// Comparing two runs, and two trials side by side.

import { api } from "./api.js";
import { ciStrip, divergingBar, hero, sliceSections, stat, table, wellKind, wellLegend, wells } from "./components.js";
import { h, pre } from "./dom.js";
import { enc, groupBy, OUTCOME_LABEL, pct, plural, points, shortId, signed, usd } from "./format.js";
import { allEvents, eventRow, stepLabel } from "./trajectory.js";

const CLASS_LABEL = {
  regressed: "Regressed",
  improved: "Improved",
  changed: "Changed",
  flaky: "Flaky",
  stable_pass: "Stable pass",
  stable_fail: "Stable fail",
};
const VERDICT = {
  REGRESSION: { label: "Regression", glyph: "▼" },
  IMPROVEMENT: { label: "Improvement", glyph: "▲" },
  NO_SIGNIFICANT_CHANGE: { label: "No significant change", glyph: "●" },
};

function pickPair(task) {
  const prefer = (trials, outcome) => trials.find((trial) => trial.outcome === outcome) || trials[0];
  if (task.delta < 0) return [prefer(task.baseline_trials, "pass"), prefer(task.candidate_trials, "fail")];
  if (task.delta > 0) return [prefer(task.baseline_trials, "fail"), prefer(task.candidate_trials, "pass")];
  return [task.baseline_trials[0], task.candidate_trials[0]];
}

export async function compareView(baselineId, candidateId) {
  const result = await api(`/compare?base=${enc(baselineId)}&cand=${enc(candidateId)}`);
  const { baseline, candidate } = result;
  const verdict = VERDICT[result.verdict];
  const [low, high] = result.delta_ci_95;
  const order = Object.keys(CLASS_LABEL);
  const tasks = [...result.tasks].sort((a, b) => order.indexOf(a.class) - order.indexOf(b.class) || a.delta - b.delta);
  const rows = tasks.map((task) => {
    const [left, right] = pickPair(task);
    const name = task.critical ? [task.task_id, h("span", { class: "muted" }, " (critical)")] : task.task_id;
    return h(
      "tr",
      {},
      h("td", {}, name),
      h("td", {}, h("span", { class: `class-tag ${task.class}` }, CLASS_LABEL[task.class])),
      h("td", {}, wells(task.baseline_trials, task.task_id)),
      h("td", {}, wells(task.candidate_trials, task.task_id)),
      h("td", {}, h("span", { class: "bar-cell" }, divergingBar(task.delta), points(task.delta))),
      h("td", { class: "num" }, task.p_worse.toFixed(3)),
      h("td", {}, h("a", { href: `#/side/${enc(left.trial_id)}/${enc(right.trial_id)}` }, "Side by side")),
    );
  });
  const deltas = result.metric_deltas;
  return [
    h("h1", {}, "Compare runs"),
    h(
      "p",
      { class: "lede" },
      "Candidate ",
      h("a", { href: `#/runs/${enc(candidate.run_id)}` }, shortId(candidate.run_id)),
      ` (${candidate.config_name}) against baseline `,
      h("a", { href: `#/runs/${enc(baseline.run_id)}` }, shortId(baseline.run_id)),
      ` (${baseline.config_name}) over ${plural(result.shared_tasks, "shared task")}.`,
    ),
    h(
      "div",
      { class: `verdict ${result.verdict.toLowerCase()}` },
      h("span", { class: "glyph", "aria-hidden": "true" }, verdict.glyph),
      h(
        "div",
        {},
        h("strong", {}, verdict.label),
        verdictReason(result),
      ),
    ),
    h(
      "div",
      { class: "headline" },
      hero(
        "Change in pass rate",
        points(result.delta),
        `95% CI ${points(low)} to ${points(high)}`,
        ciStrip(low, high, result.delta, { min: -1, max: 1, format: (v) => `${Math.round(v * 100)}` }),
      ),
      stat("Baseline pass rate", pct(baseline.macro_pass_rate), baseline.config_name),
      stat("Candidate pass rate", pct(candidate.macro_pass_rate), candidate.config_name),
      stat("Cost per trial", signed(deltas.cost_usd, 4, "$"), `Steps ${signed(deltas.steps, 1)}, duration ${signed(deltas.duration_s, 1)} s`),
    ),
    result.warnings.map((warning) => h("div", { class: "notice" }, warning[0].toUpperCase() + warning.slice(1))),
    h("h2", {}, "Tasks"),
    wellLegend(),
    table(["Task", "Class", "Baseline", "Candidate", "Change", "p (worse)", "Trajectories"], rows, [5]),
    sliceSections(result.slices, "Mean change", (value) => h("span", { class: "bar-cell" }, divergingBar(value), points(value))),
    configDiff(result.config_diff),
    excludedSection(result.excluded),
  ];
}

function verdictReason(result) {
  const regressedCritical = result.tasks.filter((task) => task.critical && task.class === "regressed");
  if (result.verdict === "REGRESSION" && regressedCritical.length) {
    return h("div", {}, `Critical task regressed: ${regressedCritical.map((task) => task.task_id).join(", ")}.`);
  }
  if (result.verdict === "NO_SIGNIFICANT_CHANGE") {
    return h("div", {}, "The 95% confidence interval of the change includes zero.");
  }
  return h("div", {}, `The 95% confidence interval of the change lies entirely ${result.verdict === "REGRESSION" ? "below" : "above"} zero.`);
}

function configDiff(changes) {
  if (!changes.length) return [h("h2", {}, "Agent config"), h("p", { class: "muted" }, "Both runs used the same agent config.")];
  const value = (v) => h("code", {}, v === null || v === undefined ? "(not set)" : typeof v === "string" ? v : JSON.stringify(v));
  return [
    h("h2", {}, "Agent config changes"),
    h(
      "div",
      { class: "diff-list" },
      changes.map((change) =>
        h(
          "div",
          { class: "change" },
          h("strong", {}, change.field),
          change.diff
            ? pre(change.diff.join("\n"))
            : [h("div", { class: "before" }, h("span", { class: "muted" }, "Baseline"), value(change.baseline)), h("div", { class: "after" }, h("span", { class: "muted" }, "Candidate"), value(change.candidate))],
        ),
      ),
    ),
  ];
}

function excludedSection(excluded) {
  const names = {
    only_in_baseline: "Only in the baseline",
    only_in_candidate: "Only in the candidate",
    version_drift: "Different task versions (compare with --allow-version-drift)",
    unscored: "No scored trials in one of the runs",
  };
  const groups = Object.entries(excluded).filter(([, ids]) => ids.length);
  if (!groups.length) return null;
  return [
    h("h2", {}, "Tasks left out of the comparison"),
    table(["Reason", "Tasks"], groups.map(([key, ids]) => h("tr", {}, h("td", {}, names[key] || key), h("td", { class: "wrap" }, ids.join(", "))))),
  ];
}

export async function sideView(leftId, rightId) {
  const [left, right] = await Promise.all([api(`/trials/${enc(leftId)}`), api(`/trials/${enc(rightId)}`)]);
  const [leftEvents, rightEvents] = await Promise.all([allEvents(left.trial_id, left.attempt), allEvents(right.trial_id, right.attempt)]);
  const leftSteps = groupBy(leftEvents, "step");
  const rightSteps = groupBy(rightEvents, "step");
  const steps = [...new Set([...leftSteps.keys(), ...rightSteps.keys()])].sort((a, b) => a - b);
  const column = (trial) =>
    h(
      "div",
      {},
      h("h2", {}, h("a", { href: `#/trials/${enc(trial.trial_id)}` }, `Trial ${trial.trial_index + 1}`), ` of run ${shortId(trial.run_id)}`),
      h("p", {}, h("span", { class: `outcome ${trial.outcome}` }, OUTCOME_LABEL[wellKind(trial)]), trial.termination ? `, ${trial.termination}` : "", `, ${trial.steps ?? 0} steps, ${usd(trial.cost_usd)}`),
    );
  const cell = (events, step) =>
    h("div", { class: "timeline" }, h("div", { class: "step-head" }, stepLabel(step)), events ? events.map(eventRow) : h("div", { class: "side-empty" }, "No events at this step."));
  return [
    h("h1", {}, `${left.task_id}, side by side`),
    h("p", { class: "lede" }, "Steps line up across the two trials, so you can see where their paths split."),
    h("div", { class: "side" }, column(left), column(right), steps.map((step) => [cell(leftSteps.get(step), step), cell(rightSteps.get(step), step)])),
  ];
}
