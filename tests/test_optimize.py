"""Tuning an LM step's instructions against labels: `table.optimize(...)`."""

import pandas as pd
import pytest
from fakes import FIELD, FuncLM

import tracepress as tp

TRUTH = {"alpha__1": True, "alpha__2": False, "beta__1": True}


class InstructionLM(FuncLM):
    """Answers correctly only when its instructions say GOOD; otherwise always False."""

    def forward(self, prompt=None, messages=None, **kwargs):
        good = "GOOD" in messages[0]["content"]
        self.fn = lambda fields: {"doom_loop": str(TRUTH[fields["trial"].strip()] if good else False)}
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class Reflector:
    """Stands in for GEPA's reflection LM: always proposes the instructions that work."""

    def __call__(self, prompt=None, **kwargs):
        return ["```\nGOOD: decide from the steps whether this run looped.\n```"]


def detector(src, lm):
    return src.steps.group_by("trial").lm_reduce("trial, message -> doom_loop: bool", lm=lm,
                                                instructions="Did the agent loop?")


def test_optimize_finds_better_instructions(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    lm = InstructionLM(lambda f: {})
    labels = pd.DataFrame({"trial": list(TRUTH), "doom_loop": list(TRUTH.values())})
    tuned = detector(src, lm).optimize(labels, metric="doom_loop", budget=20, reflection_lm=Reflector())
    assert "GOOD" in tuned.instructions
    report = tuned.optimize_report
    assert report.before == "Did the agent loop?" and report.after == tuned.instructions
    assert report.train_after == 1.0 and report.val_after >= report.val_before
    assert {r.trial: r.doom_loop for r in tuned.collect()} == TRUTH  # the tuned step runs the new prompt


def test_optimize_with_a_metric_function(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    labels = [{"trial": t, "doom_loop": v} for t, v in TRUTH.items()]
    metric = lambda row, label: (float(row.doom_loop == label["doom_loop"]), "compared")
    tuned = detector(src, InstructionLM(lambda f: {})).optimize(labels, metric, budget=20, reflection_lm=Reflector())
    assert tuned.optimize_report.train_after == 1.0


def test_optimize_errors(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    labels = pd.DataFrame({"trial": list(TRUTH), "doom_loop": list(TRUTH.values())})
    with pytest.raises(ValueError, match="isn't an LM step"):
        src.trials.optimize(labels, metric="reward")
    step = detector(src, InstructionLM(lambda f: {}))
    with pytest.raises(ValueError, match="need a 'trial' column"):
        step.optimize(pd.DataFrame({"task": ["bench/alpha"], "doom_loop": [True]}), metric="doom_loop")
    with pytest.raises(ValueError, match="isn't an output"):
        step.optimize(labels.rename(columns={"doom_loop": "loop"}), metric="loop")
    with pytest.raises(ValueError, match="no rows for labeled"):
        step.optimize(pd.DataFrame({"trial": ["nope"], "doom_loop": [True]}), metric="doom_loop")
