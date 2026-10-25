// The run list and one run's dashboard.

import { api } from "./api.js";
import { ciStrip, ciText, hero, magnitude, runName, sliceSections, stat, statusText, table, wellLegend, wells } from "./components.js";
import { h } from "./dom.js";
import { enc, groupBy, pct, plural, sentence, shortId, usd, when } from "./format.js";
import { hooks, render } from "./page.js";

export async function runsView() {
  const { runs } = await api("/runs");
  if (!runs.length) {
    return [
      h("h1", {}, "Runs"),
      h(
        "p",
        { class: "lede" },
        "No runs yet. Validate tasks with ",
        h("code", {}, "agentoscopy task validate --all"),
        ", then start a run with ",
        h("code", {}, "agentoscopy run --suite example --agent configs/scripted-fix.yaml"),
        ".",
      ),
    ];
  }
  const selected = new Set();
  const buttons = selectionButtons(runs, selected);
  const rows = runs.map((run) => {
    const box = h("input", { type: "checkbox", "aria-label": `Select run ${shortId(run.run_id)}` });
    box.addEventListener("change", () => {
      if (box.checked) selected.add(run.run_id);
      else selected.delete(run.run_id);
      box.closest("tr").classList.toggle("selected", box.checked);
      buttons.update();
    });
    const labels = Object.entries(run.labels || {}).map(([k, v]) => `${k}=${v}`).join(", ");
    return h(
      "tr",
      {},
      h("td", {}, box),
      h("td", {}, h("a", { href: `#/runs/${enc(run.run_id)}` }, shortId(run.run_id))),
      h("td", {}, when(run.created_at)),
      h("td", {}, runName(run)),
      h("td", {}, run.config_name),
      h("td", {}, statusText(run)),
      h("td", { class: "num" }, pct(run.macro_pass_rate)),
      h("td", { class: "num" }, ciText(run.macro_ci_95)),
      h("td", { class: "num" }, usd(run.spent_usd)),
      h("td", { class: "wrap" }, labels),
    );
  });
  hooks.refresh = () => {
    if (selected.size === 0) render(runsView, { keepFocus: true });
  };
  return [
    h("h1", {}, "Runs"),
    h("p", { class: "lede" }, "Select two runs to compare them; the earlier run is the baseline. Select two or more to weigh pass rate against cost."),
    buttons.toolbar,
    table(
      ["", "Run", "Started", "Suite", "Agent", "Status", "Pass rate", "95% CI", "Cost", "Labels"],
      rows,
      [6, 7, 8],
    ),
  ];
}

function selectionButtons(runs, selected) {
  const chosen = () => runs.filter((run) => selected.has(run.run_id));
  const compare = h("button", { class: "primary", type: "button", disabled: true }, "Compare selected runs");
  compare.addEventListener("click", () => {
    const [baseline, candidate] = chosen().sort((a, b) => a.created_at.localeCompare(b.created_at));
    location.hash = `#/compare/${enc(baseline.run_id)}/${enc(candidate.run_id)}`;
  });
  const weigh = h("button", { type: "button", disabled: true }, "Compare cost and quality");
  weigh.addEventListener("click", () => {
    location.hash = `#/tradeoff/${chosen().map((run) => enc(run.run_id)).join(",")}`;
  });
  const update = () => {
    compare.disabled = selected.size !== 2;
    weigh.disabled = selected.size < 2;
  };
  return { toolbar: h("div", { class: "toolbar" }, compare, weigh), update };
}

export async function runView(runId) {
  const [run, summary, { trials }] = await Promise.all([
    api(`/runs/${enc(runId)}`),
    api(`/runs/${enc(runId)}/summary`),
    api(`/runs/${enc(runId)}/trials`),
  ]);
  const byTask = groupBy(trials, "task_id");
  const counts = summary.counts;
  const ci = summary.macro_ci_95;
  const k = summary.k;
  const taskRows = summary.tasks.map((task) =>
    h(
      "tr",
      {},
      h("td", {}, task.task_id),
      h("td", {}, wells(byTask.get(task.task_id) || [], task.task_id)),
      h("td", {}, h("span", { class: "bar-cell" }, magnitude(task.pass_rate), pct(task.pass_rate))),
      h("td", { class: "num" }, `${task.passes} of ${task.n}`),
      h("td", { class: "num" }, pct(task.pass_at_k)),
      h("td", { class: "num" }, pct(task.pass_hat_k)),
      h("td", { class: "num" }, usd(task.mean_cost_usd)),
      h("td", { class: "num" }, task.mean_steps == null ? "n/a" : task.mean_steps.toFixed(1)),
    ),
  );
  hooks.refresh = (runs) => {
    const latest = runs.find((item) => item.run_id === runId);
    if (latest && (latest.status !== run.status || latest.finished_trials !== run.finished_trials)) {
      render(() => runView(runId), { keepFocus: true });
    }
  };
  return [
    h("h1", {}, `Run ${shortId(run.run_id)}`),
    h(
      "p",
      { class: "lede" },
      `${runName(run)} with ${run.config_name}, ${plural(run.trials_per_task, "trial")} per task, seed ${run.seed}. `,
      `Started ${when(run.created_at)}. ${statusText(run)}.`,
    ),
    cancelControl(run),
    h(
      "div",
      { class: "headline" },
      hero(
        "Macro pass rate",
        pct(summary.macro_pass_rate),
        ci ? `95% CI ${pct(ci[0])} to ${pct(ci[1])}` : "No scored trials yet",
        ci && ciStrip(ci[0], ci[1], summary.macro_pass_rate, { min: 0, max: 1, format: (v) => `${Math.round(v * 100)}%` }),
      ),
      stat(
        "Scored trials",
        String(counts.pass + counts.fail),
        `${counts.pass} pass, ${counts.fail} fail; ${counts.infra_error} infra error, ${counts.skipped + counts.cancelled} not run`,
      ),
      stat("Cost", usd(summary.total_cost_usd, 2), costNote(summary)),
      stat("Flaky tasks", String(summary.flaky_tasks.length), summary.flaky_tasks.length ? summary.flaky_tasks.join(", ") : "Every task gave consistent results"),
    ),
    summary.uncalibrated_judges?.length
      ? h(
          "div",
          { class: "notice" },
          h("strong", {}, "Uncalibrated judges. "),
          `Human reviewers often disagree with ${summary.uncalibrated_judges.join(", ")}, so pass rates that depend on them may be wrong. `,
          h("a", { href: "#/calibration" }, "See calibration"),
        )
      : null,
    h("h2", {}, "Tasks"),
    wellLegend(),
    table(["Task", "Trials", "Pass rate", "Passes", `pass@${k}`, `pass^${k}`, "Cost per trial", "Steps"], taskRows, [3, 4, 5, 6, 7]),
    sliceSections(summary.slices, "Mean pass rate", (value) => h("span", { class: "bar-cell" }, magnitude(value), pct(value))),
    tagCounts(summary.failure_tags),
  ];
}

function tagCounts(counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return null;
  return [
    h("h2", {}, "Failure tags"),
    table(["Tag", "Trials"], entries.map(([tag, n]) => h("tr", {}, h("td", {}, tag.replaceAll("_", " ")), h("td", { class: "num" }, String(n)))), [1]),
  ];
}

function cancelControl(run) {
  if (run.status !== "running" && run.status !== "pending") return null;
  const button = h("button", { type: "button" }, "Cancel run");
  const note = h("span", { class: "muted", role: "status" });
  let armed = false;
  button.addEventListener("click", async () => {
    if (!armed) {
      // A second click confirms, so a stray click never stops a run.
      armed = true;
      button.textContent = "Confirm: cancel run";
      button.classList.add("danger");
      note.textContent = "Queued trials stop at once; running ones within a few seconds. Finished trials keep their results.";
      return;
    }
    button.disabled = true;
    try {
      const result = await api(`/runs/${enc(run.run_id)}`, { method: "DELETE" });
      note.textContent = result.status === "cancelled" ? "Cancelled." : "Cancelling.";
    } catch (error) {
      note.textContent = sentence(error.message);
      button.disabled = false;
    }
  });
  return h("div", { class: "toolbar" }, button, note);
}

function costNote(summary) {
  const perPass = summary.cost_per_pass_usd == null ? "No passes" : `${usd(summary.cost_per_pass_usd)} per pass`;
  return summary.judge_cost_usd ? `${perPass}; judging ${usd(summary.judge_cost_usd)} more` : perPass;
}
