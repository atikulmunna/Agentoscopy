// Pass rate against cost across runs, and the runs no other run beats on both (the frontier).

import { api } from "./api.js";
import { attachTip, table } from "./components.js";
import { h, svg } from "./dom.js";
import { enc, pct, plural, sentence, shortId, usd } from "./format.js";

const WIDTH = 720;
const HEIGHT = 380;
const PAD = { left: 56, right: 24, top: 36, bottom: 56 };
const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;
const LABEL_ALL_UP_TO = 8; // past this, only frontier runs are named on the chart; the table names all
const LABEL_BOX = { width: 120, height: 20 }; // roughly one run name, in chart units

export async function tradeoffView(runIds) {
  const result = await api(`/tradeoff?runs=${runIds.map(enc).join(",")}`);
  const onFrontier = result.runs.filter((run) => run.on_frontier).length;
  const rows = result.runs.map((run) =>
    h(
      "tr",
      {},
      h("td", {}, h("a", { href: `#/runs/${enc(run.run_id)}` }, shortId(run.run_id))),
      h("td", {}, run.config_name),
      h("td", { class: "num" }, pct(run.pass_rate)),
      h("td", { class: "num" }, `${pct(run.pass_rate_ci_95[0])} to ${pct(run.pass_rate_ci_95[1])}`),
      h("td", { class: "num" }, usd(run.cost_per_trial_usd)),
      h("td", { class: "num" }, `${run.passes} of ${run.trials}`),
      h("td", {}, run.on_frontier ? "Yes" : "No"),
    ),
  );
  return [
    h("h1", {}, "Cost and quality"),
    h(
      "p",
      { class: "lede" },
      `${plural(result.runs.length, "run")} over ${plural(result.shared_tasks, "shared task")}; ${onFrontier} on the frontier, `,
      "where no other run passes at least as often for no more cost. Judging is not counted in the cost.",
    ),
    result.warnings.map((warning) => h("div", { class: "notice" }, sentence(warning))),
    h("div", { class: "panel scroll-x tradeoff" }, legend(), chart(result.runs)),
    h("h2", {}, "Runs, cheapest first"),
    table(["Run", "Agent", "Pass rate", "95% CI", "Cost per trial", "Passes", "Frontier"], rows, [2, 3, 4, 5]),
  ];
}

function legend() {
  return h(
    "div",
    { class: "legend" },
    h("span", {}, h("i", { class: "swatch frontier", "aria-hidden": "true" }), "On the frontier"),
    h("span", {}, h("i", { class: "swatch beaten", "aria-hidden": "true" }), "Beaten by another run"),
    h("span", {}, h("i", { class: "swatch step", "aria-hidden": "true" }), "Best pass rate for the cost"),
    h("span", {}, h("i", { class: "swatch whisker", "aria-hidden": "true" }), "95% CI"),
  );
}

function chart(runs) {
  const scale = costScale(runs);
  const frontier = runs.filter((run) => run.on_frontier);
  const beaten = runs.filter((run) => !run.on_frontier);
  const named = namedRuns(runs.length <= LABEL_ALL_UP_TO ? [...frontier, ...beaten] : frontier, scale);
  return svg(
    "svg",
    { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "group", "aria-label": "Pass rate against agent cost per trial, one point per run" },
    ...axes(scale),
    frontierStep(frontier, scale),
    ...[...beaten, ...frontier].map((run) => point(run, scale, named.has(run.run_id))),
  );
}

// Frontier runs claim label space first; a label that would land on one already placed is left
// to the tooltip and the table (two runs of one config often sit almost on top of each other).
function namedRuns(candidates, scale) {
  const placed = [];
  for (const run of candidates) {
    const x = scale.x(run.cost_per_trial_usd);
    const y = scale.y(run.pass_rate);
    const clashes = placed.some((spot) => Math.abs(spot.x - x) < LABEL_BOX.width && Math.abs(spot.y - y) < LABEL_BOX.height);
    if (!clashes) placed.push({ x, y, runId: run.run_id });
  }
  return new Set(placed.map((spot) => spot.runId));
}

function costScale(runs) {
  const top = Math.max(...runs.map((run) => run.cost_per_trial_usd)) || 0.001; // mock runs cost nothing
  const rough = top / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = magnitude * [1, 2, 5, 10].find((m) => rough <= magnitude * m);
  const max = Math.ceil(top / step - 1e-9) * step; // 0.01 / 0.002 is 5.000000000000001
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  return {
    x: (cost) => PAD.left + (cost / max) * PLOT_W,
    y: (rate) => PAD.top + (1 - rate) * PLOT_H,
    ticks: Array.from({ length: Math.round(max / step) + 1 }, (_, i) => i * step),
    format: (cost) => usd(cost, digits),
  };
}

function axes(scale) {
  const bottom = PAD.top + PLOT_H;
  const rates = [0, 0.25, 0.5, 0.75, 1];
  return [
    ...rates.map((rate) => svg("line", { class: rate ? "grid" : "baseline", x1: PAD.left, x2: PAD.left + PLOT_W, y1: scale.y(rate), y2: scale.y(rate) })),
    ...rates.map((rate) => svg("text", { class: "tick", x: PAD.left - 10, y: scale.y(rate) + 4, "text-anchor": "end" }, `${rate * 100}%`)),
    ...scale.ticks.map((cost) => svg("text", { class: "tick", x: scale.x(cost), y: bottom + 20, "text-anchor": "middle" }, scale.format(cost))),
    svg("text", { class: "title", x: PAD.left + PLOT_W / 2, y: HEIGHT - 8, "text-anchor": "middle" }, "Agent cost per trial (USD)"),
    svg("text", { class: "title", x: PAD.left - 10, y: PAD.top - 18, "text-anchor": "start" }, "Pass rate"),
  ];
}

// Between two frontier runs, the best a budget buys is the cheaper run's pass rate, so the
// line steps rather than slopes; past the last run, spending more buys nothing better.
function frontierStep(frontier, scale) {
  const [first, ...rest] = frontier;
  const moves = rest.map((run) => `H${scale.x(run.cost_per_trial_usd)}V${scale.y(run.pass_rate)}`);
  const d = `M${scale.x(first.cost_per_trial_usd)},${scale.y(first.pass_rate)}${moves.join("")}H${PAD.left + PLOT_W}`;
  return svg("path", { class: "step", d });
}

function point(run, scale, labelled) {
  const [low, high] = run.pass_rate_ci_95;
  const cx = scale.x(run.cost_per_trial_usd);
  const cy = scale.y(run.pass_rate);
  const name = run.config_name || shortId(run.run_id);
  // The step line leaves each frontier run to the right and reaches it from below, so a label
  // below and to the right stays clear of it (above, near the floor; to the left, near the edge).
  const nearRight = cx > PAD.left + PLOT_W * 0.75;
  const labelY = run.pass_rate > 0.1 ? cy + 24 : cy - 12;
  const summary = `${pct(run.pass_rate)} pass rate (95% CI ${pct(low)} to ${pct(high)}), ${usd(run.cost_per_trial_usd)} per trial`;
  const node = svg(
    "a",
    { class: `point ${run.on_frontier ? "frontier" : "beaten"}`, href: `#/runs/${enc(run.run_id)}`, "aria-label": `${name}: ${summary}` },
    svg("circle", { class: "hit", cx, cy, r: 14 }),
    svg("line", { class: "whisker", x1: cx, x2: cx, y1: scale.y(high), y2: scale.y(low) }),
    svg("circle", { class: "halo", cx, cy, r: 8 }),
    svg("circle", { class: "dot", cx, cy, r: 5 }),
    ...(labelled ? [svg("text", { class: "name", x: nearRight ? cx - 12 : cx + 12, y: labelY, "text-anchor": nearRight ? "end" : "start" }, name)] : []),
  );
  const detail = `${summary}; ${run.passes} of ${run.trials} trials passed; run ${shortId(run.run_id)}`;
  return attachTip(node, name, detail);
}
