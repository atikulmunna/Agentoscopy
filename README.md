# Agentoscopy

An evaluation harness for LLM agents. This is milestone M2: suites of tasks run as
concurrent trials in Docker sandboxes, every model call goes through a budget-enforcing
gateway, results land in SQLite, and runs can be compared statistically, from the CLI
or in a local web UI.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and a running Docker daemon.

```sh
uv sync
```

## Workflow

```sh
uv run agentoscopy task validate --all                  # build images, run null/reference/isolation checks
uv run agentoscopy task list                            # validation status of every task
uv run agentoscopy run --suite example --agent configs/scripted-fix.yaml --trials 3
uv run agentoscopy report <run_id> --format md          # table, md, or json
uv run agentoscopy compare <baseline_run> <candidate_run>
uv run agentoscopy serve                                # web UI; open the URL it prints
```

`agentoscopy run` takes `--suite NAME` or one or more `--task ID`, plus `--budget-usd` (run cost
ceiling, default 10), `--concurrency`, `--seed`, and `--label KEY=VALUE`. A task can only
run once its current content has been validated.

Exit codes: `0` success, `1` a task failed a validation check, `2` invalid input, `3`
environment or harness failure.

## Comparing runs

`agentoscopy compare` lines up the tasks both runs scored at the same version. Each task is
classified as regressed, improved, changed, flaky, stable pass, or stable fail; regressed
and improved need a one-sided Fisher's exact test at p < 0.05, which takes at least 4
trials per run (6 for tasks marked `critical`, which get those automatically). The
verdict is `REGRESSION` when the 95% bootstrap CI of the mean per-task change is below
zero or a critical task regressed, `IMPROVEMENT` when it is above zero, and
`NO_SIGNIFICANT_CHANGE` otherwise. The output also shows cost, token, step, and duration
deltas, deltas by task category and tag, and a field-level diff of the two agent configs.

## Web UI

`agentoscopy serve` serves a read-only UI on `127.0.0.1:8321`: the runs list (select two runs to
compare them), a run dashboard with one well per trial, trial pages with the event
timeline, graders, and filesystem diff, the comparison, and side-by-side trajectories.
Every API call needs the token in the printed URL, and requests for other host names are
refused. Runs started from the CLI appear live.

## Agents

Agents are Python classes named by `entrypoint` in a config file under `configs/`. They
receive a recording sandbox handle and a per-trial gateway endpoint; use it as the
Anthropic SDK's `base_url` and `api_key`.

- `scripted-fix`, `scripted-broken`, `scripted-malicious`: scripted test agents. They call
  the `mock` model, which is seeded and costs nothing real.
- `claude`: an example Claude agent (tool-use loop with `bash` and `write_file`). It needs
  `ANTHROPIC_API_KEY`. The key is handed to a separate credential-proxy process and removed
  from the `agentoscopy` process before any agent code is imported.

Once a trial's step, token, or cost budget is spent, the gateway rejects further calls with
HTTP 400 and error type `budget_exceeded_error` (`agentoscopy.adapters.base.BUDGET_EXCEEDED_ERROR`).

## Tasks

A task lives in `tasks/<id>/`:

- `task.yaml`: instructions, environment, budget, graders, and optional `category`,
  `difficulty`, `tags`, and `critical`
- `fixtures/`: copied into the workdir when the task image is built
- `hidden/`: grader-only files, mounted read-only at `/hidden` during grading only
- `reference/`: optional reference solution; its files overlay the workdir during validation

Suites live in `suites/<name>.yaml` as a name and a list of task ids. The run database,
trajectories, and artifacts are written under `.agentoscopy/`.

## Tests

```sh
uv run pytest                    # integration tests are skipped if Docker is not running
uv run pytest -m "not integration"
uv run ruff check . && uv run ruff format --check agentoscopy tests
```
