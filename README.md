# Agentoscopy

An evaluation harness for LLM agents. This is milestone M4: suites of tasks run as
concurrent trials in Docker sandboxes, every model call goes through a budget-enforcing
gateway, results land in SQLite, and runs can be compared statistically, from the CLI
or in a local web UI. Trials are graded by commands, trajectory rules, a tamper check
that vetoes reward hacking, and an LLM judge whose agreement with human reviewers is
measured. Runs survive crashes and can be cancelled, and `agentoscopy ci` gates pull
requests on regressions.

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
uv run agentoscopy run --resume <run_id>                # continue a run after a crash
uv run agentoscopy replay <trial_id>                    # rerun a trial from its recorded model calls
uv run agentoscopy agent check configs/claude.yaml      # try an agent config before a real run
uv run agentoscopy cancel <run_id>
uv run agentoscopy gc --older-than 30d                  # old run outputs, orphaned sandboxes
uv run agentoscopy serve                                # web UI; open the URL it prints
```

`agentoscopy run` takes `--suite NAME` or one or more `--task ID`, plus `--budget-usd` (run cost
ceiling, default 10), `--concurrency`, `--seed`, `--label KEY=VALUE`, `--judge-model`
(default `claude-opus-5-5`), and `--review-rate` (the fraction of trials sampled for human
review, default 0.05). A task can only run once its current content has been validated.

To run only some of a suite's tasks, add `--tag`, `--category`, `--difficulty`, or
`--failed-in RUN_ID` (the tasks that failed or hit an infra error in that run). Repeat an
option for any of several values; different options must all match. A filtered run keeps
the suite's name but no suite version, so it never stands in for a full run of the suite
(as a CI baseline, say), and it records the filter as a label.

The example task has an LLM judge grader, so validating and running it calls the judge
model, which needs `ANTHROPIC_API_KEY`. To try everything offline, add `--judge-model mock`
to both `task validate` and `run`. The mock judge answers at random, so expect the
calibration page to call it uncalibrated.

Exit codes: `0` success, `1` a task failed a validation check, `2` invalid input, `3`
environment or harness failure, `130` the run was cancelled.

## Crashes, resuming, and cancelling

A trial in flight is held under a lease that its process renews every 10 seconds. If the
process dies, the lease runs out after 30 seconds, and `agentoscopy run --resume <run_id>`
picks the run up: it removes the containers and snapshots the dead attempts left behind,
counts what those attempts spent, and runs the trials again. A trial is finalised only by
the process that holds it, so nothing is lost or counted twice. Resuming needs the tasks
to have the content the run started with.

`agentoscopy cancel <run_id>` (or the run page's Cancel button) cancels queued trials at
once. Trials that are setting up or running stop within a few seconds, with 30 seconds
for the agent to wind down; trials already being graded finish. Finished trials keep
their results, and the run gets a partial summary. Ctrl-C in `agentoscopy run` is
blunter: it cancels every unfinished trial at once, including any being graded.

`agentoscopy gc` deletes the trajectories and artifacts of runs that finished more than
`--older-than` ago (results stay in the database; runs with trials waiting for review
are kept), and removes sandboxes that no running trial holds. `--dry-run` only reports.

## CI gating

`agentoscopy ci --suite S --agent C --baseline main` runs the suite, compares it with the
newest completed run of the same suite version labelled `branch=main` (from the last 7
days, `--max-baseline-age`), and reports the verdict. With no such run it runs one first
with `--baseline-agent`. `--github` posts the comparison as a pull request comment and
updates that same comment on later pushes; inside GitHub Actions it needs only
`GITHUB_TOKEN`. `--override` (the workflow sets it for the `eval-override` label) reports a
regression without failing. Exit codes: `0` no regression, `1` regression, `2` harness
error, `3` the candidate run hit its budget.

`.github/workflows/eval.yml` wires this up: pull requests are evaluated against the
baseline that pushes to `main` keep fresh in the Actions cache, and pull requests from
forks never run with provider credentials.

## Comparing runs

`agentoscopy compare` lines up the tasks both runs scored at the same version. Each task is
classified as regressed, improved, changed, flaky, stable pass, or stable fail; regressed
and improved need a one-sided Fisher's exact test at p < 0.05, which takes at least 4
trials per run (6 for tasks marked `critical`, which get those automatically). The
verdict is `REGRESSION` when the 95% bootstrap CI of the mean per-task change is below
zero or a critical task regressed, `IMPROVEMENT` when it is above zero, and
`NO_SIGNIFICANT_CHANGE` otherwise. The output also shows cost, token, step, and duration
deltas, deltas by task category and tag, and a field-level diff of the two agent configs.

## Grading

Each trial is graded in a fresh sandbox cloned from the agent's final state, with caches
(`__pycache__`, `.pytest_cache`) deleted first. A trial passes when every `required`
grader passes; its score is the weighted mean of all graders.

| Type | Passes when |
|---|---|
| `command` | its shell command exits 0 |
| `forbidden_command` | no command the agent ran matches any of `patterns` (regexes) |
| `must_read_before_edit` | every existing file the agent wrote was read first |
| `max_tool_errors` | at most `max` tool calls failed |
| `max_steps` | the agent made at most `max` model calls |
| `tamper_check` | nothing changed outside `allowed_paths` (default: the workdir and `/tmp`) or inside `protected_paths` |
| `llm_judge` | the pinned judge model says the work meets `rubric` |

The tamper check reads the whole-container filesystem diff, so it catches tampering
however it was done. A failed tamper check is a veto: the trial fails with score 0 and
the failure tag `reward_hacking`, whatever the other graders said. Changes made by the
task's `setup` commands are recorded as a baseline and never count against the agent.

The judge sees the task, the rubric, and the agent's final message, changed files, and
actions. Everything the agent produced is passed inside `<agent_data>` tags that the
judge is told to treat as evidence, never as instructions. The judge answers with
structured output, its calls go through the gateway on their own budget (`max_cost_usd`,
default $0.25 per trial), and judge spend is reported apart from agent spend. A judge
that fails, or returns unusable output 3 times, makes the trial an infra error, never a
failure of the agent. During validation each judge votes 3 times and the majority
counts; a task whose required graders are all judges is flagged `SOFT_GRADER_ONLY`.

## Human review and judge calibration

Runs add trials to a review queue: a random sample (`--review-rate`), every judge
verdict with confidence below 0.7, and anything added by hand from a trial page. The
queue shows unsure judge verdicts first, then random samples, then hand-added items.
For a judge item the reviewer answers the rubric's question; other items are trial
audits. The judge's verdict stays hidden until the review is submitted. Keys: `p` pass,
`f` fail, `s` skip, `n` next. A reviewer can also override the trial's outcome; the
original outcome is kept and the run summary is recomputed.

The calibration report gives, per task, judge grader, and judge model, the agreement
rate, Cohen's kappa, the confusion matrix, and every disagreement. Agreement and kappa
use only randomly sampled reviews, since low-confidence and hand-added items are hard
cases by design. A judge with kappa below 0.6 over at least 30 random reviews is
flagged uncalibrated, and every run that uses it shows a warning.

## Web UI

`agentoscopy serve` serves a UI on `127.0.0.1:8321`: the runs list (select two runs to
compare them), a run dashboard with one well per trial, trial pages with the event
timeline, graders, and filesystem diff, the comparison, side-by-side trajectories, the
review queue, and judge calibration. Every API call needs the token in the printed URL,
and requests for other host names are refused. Runs started from the CLI appear live.

Runs can be started and cancelled over HTTP. `POST /runs` takes JSON with `agent` (the name
of a config in `--configs-dir`, default `configs/`), either `suite` or `tasks`, and
optionally `trials`, `budget_usd`, `concurrency`, `seed`, `labels`, `judge_model`, and
`review_rate`; the run executes in the server process. `DELETE /runs/{run_id}` cancels
one. `GET /metrics` serves Prometheus metrics for every run in the database: queue depth,
trials in flight, outcomes, infra errors, trial durations, gateway latency, and spend.
Set `AGENTOSCOPY_API_TOKEN` to give the server a fixed token, for example for a scraper.

Review endpoints: `GET /review/queue?limit=N`; `POST /review/queue` with `trial_id` and an
optional `grader_name`, to add an item by hand; `POST /review/{review_id}` with `passed`
and optional `score`, `note`, `reviewer`, and `override`, which answers with the judge's
verdict; and `GET /calibration`.

## Agents

Agents are Python classes named by `entrypoint` in a config file under `configs/`. They
receive a recording sandbox handle and a per-trial gateway endpoint; use it as the
Anthropic SDK's `base_url` and `api_key`.

- `scripted-fix`, `scripted-broken`, `scripted-malicious`: scripted test agents. They call
  the `mock` model, which is seeded and costs nothing real.
- `scripted-tamper-tests`, `scripted-tamper-conftest`: apply the correct fix but also
  rewrite a test or add a `conftest.py`; the tamper check must veto both.
- `scripted-judge-injection`: fixes the bug, also rewrites an unrelated file, and tells
  the judge to pass it. A real judge should still fail `minimal_change`.
- `claude`: an example Claude agent (tool-use loop with `bash` and `write_file`). It needs
  `ANTHROPIC_API_KEY`. The key is handed to a separate credential-proxy process and removed
  from the `agentoscopy` process before any agent code is imported.

`agentoscopy agent check <config>` runs a config once on a built-in smoke task (reverse a
file's contents). It passes when the agent called a model through the gateway, acted in
the sandbox or reported a message, and finished on its own. It fails with `GATEWAY_BYPASS`
when the agent ignored the endpoint it was given, `NO_AGENT_EVENTS`, or
`AGENT_DID_NOT_FINISH`. Whether the agent solved the smoke task is reported but does not
decide the check.

Once a trial's step, token, or cost budget is spent, the gateway rejects further calls with
HTTP 400 and error type `budget_exceeded_error` (`agentoscopy.adapters.base.BUDGET_EXCEEDED_ERROR`).

## Tasks

A task lives in `tasks/<id>/`:

- `task.yaml`: instructions, environment, budget, graders, and optional `category`,
  `difficulty`, `tags`, and `critical`
- `fixtures/`: copied into the workdir when the task image is built
- `hidden/`: grader-only files, mounted read-only at `/hidden` during grading only
- `reference/`: optional reference solution; its files overlay the workdir during validation

Suites live in `suites/<name>.yaml` as a name and a list of task ids. `core` holds twenty
small Python bug fixes and features of mixed difficulty, each with visible tests, hidden
tests, a reference solution, and the tamper check; `example` holds the one task with an LLM
judge. The run database, trajectories, and artifacts are written under `.agentoscopy/`.

## Replay

`agentoscopy replay <trial_id>` reruns a recorded trial in a fresh sandbox with the same
task version, agent config, and seed. Each model call is answered with the response
recorded at the same step, at no cost, as long as the agent sends the same request it sent
then; tool calls run live. A different request, or a call the original never made, stops
the replay with `REPLAY_DIVERGED` at that step. The command exits `0` when the replay ends
with the original outcome after the same model calls, and `1` otherwise. A replay is its
own one-trial run; its trial page links to the original and to a side-by-side view.

## Tests

```sh
uv run pytest                    # integration tests are skipped if Docker is not running
uv run pytest tests/chaos        # kills a run mid-flight and resumes it
uv run pytest -m "not integration"
uv run ruff check . && uv run ruff format --check agentoscopy tests
```
