def enabled(trace):
    return trace if trace.get("enabled") else None


def split_samples(trace, ctx):
    for index, value in enumerate(trace["samples"], start=1):
        ctx.emit(
            {
                "id": f"{trace['id']}:sample-{index}",
                "trace_id": trace["id"],
                "value": value,
            }
        )


def interesting(sample):
    return sample if abs(sample["value"]) >= 4 else None


def score(sample):
    if sample["value"] < 0:
        raise ValueError("negative samples cannot be scored yet")
    return {**sample, "score": sample["value"] ** 0.5}


def summarize_trace(samples, ctx):
    return {
        "trace_id": ctx.parent["id"],
        "sample_count": len(samples),
        "scores": [sample["score"] for sample in samples],
        "failures": [failure.record_identity for failure in ctx.failures],
    }

