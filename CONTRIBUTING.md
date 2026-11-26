# Contributing

TracePress welcomes bug reports, fixes and new adapters.

**You must understand the code you contribute.** Use coding agents as much as you like, but be
ready to explain every line of your change. Harbor follows the same
[golden rule](https://github.com/harbor-framework/harbor/blob/main/CONTRIBUTING.md#golden-rule).

## What to send

- **Bugs:** open an [issue](https://github.com/Ziems/tracepress/issues/new/choose) with a minimal
  repro, or send a PR with the fix and a test.
- **New trace formats:** send a PR. The `tracepress-add-adapter` skill in
  [skills/](https://github.com/Ziems/tracepress/tree/main/skills) walks through writing an adapter,
  and [Planned adapters](https://github.com/Ziems/tracepress/blob/main/docs/SPEC.md#planned-adapters)
  lists the ones we want.
- **Changes to operators, the public API or the engine:** open an issue first so we can agree on
  the design before you write the code.
- **Security problems:** don't open an issue. Follow [SECURITY.md](SECURITY.md).
- **Small first tasks:** see
  [good first issue](https://github.com/Ziems/tracepress/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).

## Set up

Fork the repo, clone your fork, and install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Then:

```bash
uv sync
uv run pytest
```

The tests run offline against a fake LM (`tests/fakes.py`), so they need no API key.

## Make the change

- Match the style of the code around your change.
- Add a test for each fix or feature.
- If you change the public API, update the README, `docs/` and the skills in `skills/` in the same PR.
- Don't bump the version in `pyproject.toml`. The maintainer does that when cutting a release.

## Open a pull request

- Keep each PR to one change.
- Write the title as a short imperative sentence, such as "Read regraded Harbor trials'
  model and dataset". We squash-merge, so the title becomes the commit message.
- Say what changes and why, and link the issue if there is one.
- Turn on **Allow edits by maintainers**, so we can push small fixes to your branch.
