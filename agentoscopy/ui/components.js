// Pieces several views share: trial wells, charts, stats, tables, and run descriptions.

import { h, svg } from "./dom.js";
import { enc, OUTCOME_LABEL, pct } from "./format.js";

const tooltip = document.getElementById("tooltip");

export function attachTip(node, value, detail) {
  const show = () => {
    tooltip.replaceChildren(h("strong", {}, value), detail);
    tooltip.hidden = false;
    const rect = node.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const below = rect.bottom + 8 + tip.height < innerHeight;
    tooltip.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - tip.width - 8))}px`;
    tooltip.style.top = `${below ? rect.bottom + 8 : rect.top - tip.height - 8}px`;
  };
  node.addEventListener("pointerenter", show);
  node.addEventListener("focus", show);
  node.addEventListener("pointerleave", hideTooltip);
  node.addEventListener("blur", hideTooltip);
  return node;
}

export function hideTooltip() {
  tooltip.hidden = true;
}

export function wellKind(trial) {
  if (trial.outcome) return trial.outcome;
  return trial.state === "QUEUED" ? "pending" : "running";
}

function well(trial, taskId) {
  const kind = wellKind(trial);
  const label = OUTCOME_LABEL[kind] || kind;
  const number = trial.trial_index + 1;
  const node = h("a", {
    class: `well ${kind}`,
    href: `#/trials/${enc(trial.trial_id)}`,
    "aria-label": `${taskId}, trial ${number}: ${label}`,
  });
  const detail = `${taskId}, trial ${number}${trial.termination ? `, ${trial.termination}` : ""}`;
  return attachTip(node, label, detail);
}

export function wells(trials, taskId) {
  const sorted = [...trials].sort((a, b) => a.trial_index - b.trial_index);
  return h("span", { class: "wells" }, sorted.map((trial) => well(trial, taskId)));
}

export function wellLegend() {
  const keys = [
    ["pass", "Pass"],
    ["fail", "Fail"],
    ["infra_error", "Infra error (not scored)"],
    ["skipped", "Skipped or cancelled"],
    ["pending", "Not finished"],
  ];
  return h(
    "div",
    { class: "legend" },
    keys.map(([kind, label]) => h("span", {}, h("i", { class: `well ${kind}`, "aria-hidden": "true" }), label)),
  );
}

export function magnitude(rate) {
  return h("span", { class: "mag", "aria-hidden": "true" }, h("i", { style: `width:${(rate || 0) * 100}%` }));
}

export function divergingBar(delta) {
  const bar = delta === 0 ? null : h("i", { class: delta < 0 ? "neg" : "pos", style: `width:${Math.abs(delta) * 50}%` });
  return h("span", { class: "div", "aria-hidden": "true" }, bar);
}

export function ciStrip(low, high, point, { min, max, format }) {
  const left = 8;
  const width = 344;
  const x = (v) => left + ((Math.min(Math.max(v, min), max) - min) / (max - min)) * width;
  const ticks = min < 0 && max > 0 ? [min, 0, max] : [min, max];
  const chart = svg(
    "svg",
    { viewBox: "0 0 360 34", role: "img", "aria-label": `95% confidence interval ${format(low)} to ${format(high)}` },
    svg("line", { class: "axis", x1: left, x2: left + width, y1: 12, y2: 12 }),
    ...ticks.map((t) => svg("text", { x: x(t), y: 31, "text-anchor": "middle" }, format(t))),
    ...(ticks.includes(0) ? [svg("line", { class: "zero", x1: x(0), x2: x(0), y1: 4, y2: 20 })] : []),
    svg("line", { class: "range", x1: x(low), x2: x(high), y1: 12, y2: 12 }),
    svg("circle", { class: "point", cx: x(point), cy: 12, r: 5 }),
  );
  return h("div", { class: "ci" }, chart);
}

export function stat(label, value, note, extra) {
  return h("div", {}, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value" }, value), note && h("div", { class: "stat-note" }, note), extra);
}

export function hero(label, value, note, extra) {
  return h("div", {}, h("div", { class: "stat-label" }, label), h("div", { class: "hero-value" }, value), h("div", { class: "stat-note" }, note), extra);
}

export function table(headers, rows, numeric = []) {
  return h(
    "div",
    { class: "panel scroll-x" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, headers.map((label, i) => h("th", { class: numeric.includes(i) ? "num" : null, scope: "col" }, label)))),
      h("tbody", {}, rows),
    ),
  );
}

export function errorState(error) {
  if (error.status === 401) {
    return h(
      "div",
      { class: "error-state" },
      h("strong", {}, "This page needs the token from agentoscopy serve."),
      h("p", {}, "Open the link that agentoscopy serve printed in your terminal; it carries the token."),
    );
  }
  return h("div", { class: "error-state" }, h("strong", {}, error.code || "Request failed"), h("p", {}, error.message));
}

export function runName(run) {
  if (run.mode === "replay") return "Replay";
  if (run.mode === "fork") return "Fork";
  if (!run.suite_id) return "Ad hoc tasks";
  return run.suite_version == null ? `${run.suite_id}, filtered` : `${run.suite_id} v${run.suite_version}`;
}

export function statusText(run) {
  const flags = run.flags?.length ? `, ${run.flags.join(", ").toLowerCase().replaceAll("_", " ")}` : "";
  const active = { running: "Running", pending: "Pending", cancelling: "Cancelling" }[run.status];
  if (active) return `${active}, ${run.finished_trials} of ${run.total_trials}`;
  return `${run.status[0].toUpperCase()}${run.status.slice(1)}${flags}`;
}

export function ciText(ci) {
  return ci ? `${pct(ci[0])} to ${pct(ci[1])}` : "n/a";
}

export function sliceSections(slices, valueLabel, renderValue) {
  const names = { category: "category", difficulty: "difficulty", tag: "tag" };
  return Object.entries(slices || {})
    .filter(([, groups]) => groups.length > 1)
    .map(([dimension, groups]) => [
      h("h2", {}, `By ${names[dimension] || dimension}`),
      table(
        [names[dimension] || dimension, "Tasks", valueLabel],
        groups.map((group) => h("tr", {}, h("td", {}, group.value), h("td", { class: "num" }, String(group.tasks)), h("td", {}, renderValue(group.mean)))),
        [1],
      ),
    ]);
}
