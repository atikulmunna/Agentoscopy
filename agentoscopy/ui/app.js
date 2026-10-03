// Agentoscopy web UI: hash-routed views over the REST API.
// Everything shown comes from runs and agents, so it is untrusted text: values reach the DOM
// through textContent (the `h` helper), never through innerHTML.

const TOKEN_KEY = "agentoscopy.token";
const THEME_KEY = "agentoscopy.theme";
const THEMES = ["system", "light", "dark"];
const PAGE_SIZE = 2000;
const OUTCOME_LABEL = {
  pass: "Pass",
  fail: "Fail",
  infra_error: "Infra error",
  skipped: "Skipped",
  cancelled: "Cancelled",
  pending: "Queued",
  running: "Running",
};
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
const EVENT_GROUPS = {
  all: null,
  model: ["model_request", "model_response"],
  tools: ["tool_call", "tool_result"],
  messages: ["agent_message"],
  problems: ["error", "budget_warning"],
};
const KIND_LABEL = {
  deterministic: "Command",
  trajectory: "Trajectory",
  tamper: "Tamper check",
  judge: "LLM judge",
  custom: "Custom",
};
const SOURCE_LABEL = {
  low_confidence: "Low judge confidence",
  random: "Random sample",
  manual: "Added by hand",
};
const CALIBRATION_LABEL = {
  calibrated: "Calibrated",
  uncalibrated: "Uncalibrated",
  insufficient_data: "Needs more reviews",
  undefined: "Undefined",
};

const view = document.getElementById("view");
const tooltip = document.getElementById("tooltip");
let token = stored(TOKEN_KEY);
let refreshCurrent = null; // set by a view that wants live updates
let keyHandler = null; // set by a view that takes single-key shortcuts
const skippedReviews = new Set(); // review items skipped during this visit

// ---------------------------------------------------------------- basics

function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null; // storage can be unavailable; the page still works for this visit
  }
}

function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // remembering is a convenience only
  }
}

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

async function api(path, { method = "GET", body: payload } = {}) {
  const headers = { Authorization: `Bearer ${token || ""}` };
  if (payload !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { method, headers, body: payload === undefined ? undefined : JSON.stringify(payload) });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error?.message || `The server answered ${response.status}.`);
    error.code = body.error?.code;
    error.status = response.status;
    throw error;
  }
  return body;
}

const enc = encodeURIComponent;
const pct = (v) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);
const sign = (v) => (v < 0 ? "-" : "+");
const points = (v) => (v == null ? "n/a" : `${sign(v)}${Math.abs(v * 100).toFixed(1)} points`);
const usd = (v, digits = 4) => (v == null ? "n/a" : `$${v.toFixed(digits)}`);
const signed = (v, digits, prefix = "") => (v == null ? "n/a" : `${sign(v)}${prefix}${Math.abs(v).toFixed(digits)}`);
const shortId = (id) => String(id).slice(0, 8);
const when = (iso) => (iso ? new Date(iso).toLocaleString() : "");
const json = (value) => JSON.stringify(value, null, 2);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const verdictText = (passed) => (passed ? "Pass" : "Fail");
const sentence = (text) => `${text[0].toUpperCase()}${text.slice(1)}${/[.!?]$/.test(text) ? "" : "."}`;

function groupBy(items, key) {
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item[key])) groups.set(item[key], []);
    groups.get(item[key]).push(item);
  }
  return groups;
}

// ---------------------------------------------------------------- shared pieces

function attachTip(node, value, detail) {
  const show = () => {
    tooltip.replaceChildren(h("strong", {}, value), detail);
    tooltip.hidden = false;
    const rect = node.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const below = rect.bottom + 8 + tip.height < innerHeight;
    tooltip.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - tip.width - 8))}px`;
    tooltip.style.top = `${below ? rect.bottom + 8 : rect.top - tip.height - 8}px`;
  };
  const hide = () => {
    tooltip.hidden = true;
  };
  node.addEventListener("pointerenter", show);
  node.addEventListener("focus", show);
  node.addEventListener("pointerleave", hide);
  node.addEventListener("blur", hide);
  return node;
}

function wellKind(trial) {
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

function wells(trials, taskId) {
  const sorted = [...trials].sort((a, b) => a.trial_index - b.trial_index);
  return h("span", { class: "wells" }, sorted.map((trial) => well(trial, taskId)));
}

function wellLegend() {
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

function magnitude(rate) {
  return h("span", { class: "mag", "aria-hidden": "true" }, h("i", { style: `width:${(rate || 0) * 100}%` }));
}

function divergingBar(delta) {
  const bar = delta === 0 ? null : h("i", { class: delta < 0 ? "neg" : "pos", style: `width:${Math.abs(delta) * 50}%` });
  return h("span", { class: "div", "aria-hidden": "true" }, bar);
}

function ciStrip(low, high, point, { min, max, format }) {
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

function stat(label, value, note, extra) {
  return h("div", {}, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value" }, value), note && h("div", { class: "stat-note" }, note), extra);
}

function hero(label, value, note, extra) {
  return h("div", {}, h("div", { class: "stat-label" }, label), h("div", { class: "hero-value" }, value), h("div", { class: "stat-note" }, note), extra);
}

function table(headers, rows, numeric = []) {
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

function errorState(error) {
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

function runName(run) {
  return run.suite_id ? `${run.suite_id} v${run.suite_version}` : "Ad hoc tasks";
}

function statusText(run) {
  const flags = run.flags?.length ? `, ${run.flags.join(", ").toLowerCase().replaceAll("_", " ")}` : "";
  const active = { running: "Running", pending: "Pending", cancelling: "Cancelling" }[run.status];
  if (active) return `${active}, ${run.finished_trials} of ${run.total_trials}`;
  return `${run.status[0].toUpperCase()}${run.status.slice(1)}${flags}`;
}

function ciText(ci) {
  return ci ? `${pct(ci[0])} to ${pct(ci[1])}` : "n/a";
}

// ---------------------------------------------------------------- runs

async function runsView() {
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
  const compare = h("button", { class: "primary", type: "button", disabled: true }, "Compare selected runs");
  compare.addEventListener("click", () => {
    const [baseline, candidate] = runs
      .filter((run) => selected.has(run.run_id))
      .sort((a, b) => a.created_at.localeCompare(b.created_at));
    location.hash = `#/compare/${enc(baseline.run_id)}/${enc(candidate.run_id)}`;
  });
  const rows = runs.map((run) => {
    const box = h("input", { type: "checkbox", "aria-label": `Select run ${shortId(run.run_id)}` });
    box.addEventListener("change", () => {
      if (box.checked) selected.add(run.run_id);
      else selected.delete(run.run_id);
      box.closest("tr").classList.toggle("selected", box.checked);
      compare.disabled = selected.size !== 2;
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
  refreshCurrent = () => {
    if (selected.size === 0) render(runsView, { keepFocus: true });
  };
  return [
    h("h1", {}, "Runs"),
    h("p", { class: "lede" }, "Select two runs to compare them. The earlier run is the baseline."),
    h("div", { class: "toolbar" }, compare),
    table(
      ["", "Run", "Started", "Suite", "Agent", "Status", "Pass rate", "95% CI", "Cost", "Labels"],
      rows,
      [6, 7, 8],
    ),
  ];
}

// ---------------------------------------------------------------- one run

async function runView(runId) {
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
  refreshCurrent = (runs) => {
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

function sliceSections(slices, valueLabel, renderValue) {
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

// ---------------------------------------------------------------- trajectories

async function allEvents(trialId, attempt) {
  const events = [];
  for (;;) {
    const page = await api(`/trials/${enc(trialId)}/trajectory?attempt=${attempt}&offset=${events.length}&limit=${PAGE_SIZE}`);
    events.push(...page.events);
    if (!page.events.length || events.length >= page.total) return events;
  }
}

function pre(text) {
  return h("pre", {}, text);
}

function section(title, ...content) {
  return [h("h4", {}, title), ...content];
}

function contentSummary(content) {
  for (const block of content || []) {
    if (block.type === "text" && block.text) return block.text.split("\n")[0];
  }
  const tools = (content || []).filter((block) => block.type === "tool_use").map((block) => block.name);
  return tools.length ? `Calls ${tools.join(", ")}` : "No text";
}

function contentBody(content) {
  return (content || []).map((block) => {
    if (block.type === "text") return section("Text", pre(block.text));
    if (block.type === "tool_use") return section(`Tool call: ${block.name}`, pre(json(block.input ?? block.input_raw)));
    if (block.type === "thinking") return block.thinking ? section("Thinking", pre(block.thinking)) : null;
    return section(block.type, pre(json(block)));
  });
}

function describe(event) {
  const p = event.payload || {};
  const raw = () => [pre(json(p))];
  switch (event.type) {
    case "trial_start":
      return { kind: "Start", text: `Task ${p.task_id}, agent ${p.config_name}`, body: raw };
    case "model_request":
      return {
        kind: "Model call",
        text: `${p.model}, up to ${p.params?.max_tokens} tokens${p.tools?.length ? `, tools ${p.tools.join(", ")}` : ""}`,
        body: () => [...section("Parameters", pre(json(p.params))), ...section("Request hash", pre(p.request_hash))],
      };
    case "model_response":
      if (p.status) return { kind: "Model error", text: `Provider answered ${p.status}`, body: () => section("Error", pre(p.error)) };
      return {
        kind: "Model reply",
        text: contentSummary(p.content),
        meta: `${p.output_tokens ?? 0} out, ${usd(p.cost_usd, 5)}`,
        body: () => [
          ...contentBody(p.content),
          ...section("Usage", pre(json({ stop_reason: p.stop_reason, input_tokens: p.input_tokens, output_tokens: p.output_tokens, cache_read_tokens: p.cache_read_tokens, cache_write_tokens: p.cache_write_tokens, cost_usd: p.cost_usd, latency_ms: p.latency_ms, backoff_ms: p.backoff_ms }))),
        ],
      };
    case "tool_call": {
      const args = p.args || {};
      if (p.tool === "exec") return { kind: "Command", text: args.cmd, code: true, body: () => section("Command", pre(args.cmd)) };
      if (p.tool === "write_file") return { kind: "Write file", text: `${args.path} (${args.bytes} bytes)`, body: () => section("Content", pre(args.content)) };
      return { kind: "Read file", text: args.path, body: raw };
    }
    case "tool_result": {
      const status = p.error ? `Failed: ${p.error}` : p.timed_out ? "Timed out" : `Exit ${p.exit_code}`;
      return {
        kind: "Result",
        text: status,
        meta: p.duration_ms == null ? "" : `${p.duration_ms} ms`,
        body: () => [
          p.stdout ? section("Output", pre(p.stdout)) : null,
          p.stderr ? section("Errors", pre(p.stderr)) : null,
          p.output_ref ? section("Full output", pre(`Stored as ${p.output_ref} in the trial's artifacts`)) : null,
          !p.stdout && !p.stderr ? section("Output", pre("(none)")) : null,
        ],
      };
    }
    case "agent_message":
      return { kind: "Agent", text: p.text, body: () => section("Message", pre(p.text)) };
    case "budget_warning":
      return { kind: "Budget", text: `${p.dimension} budget spent: ${p.used} of ${p.limit}`, body: raw };
    case "error":
      return { kind: "Error", text: `${p.source}: ${p.message}`, body: () => section("Message", pre(p.message)) };
    case "trial_end":
      return { kind: "End", text: `${OUTCOME_LABEL[p.outcome] || p.outcome}${p.termination ? `, ${p.termination}` : ""}`, body: raw };
    default:
      return { kind: event.type, text: JSON.stringify(p), body: raw };
  }
}

function eventRow(event) {
  const info = describe(event);
  const body = h("div", { class: "event-body" });
  const row = h(
    "details",
    { class: `event ${event.type}` },
    h(
      "summary",
      {},
      h("span", { class: "seq" }, `#${event.seq}`),
      h("span", { class: "kind" }, info.kind),
      h("span", { class: info.code ? "text code" : "text", title: info.text }, info.text ?? ""),
      h("span", { class: "meta" }, info.meta || ""),
    ),
    body,
  );
  // Bodies are built on first open, so long trajectories render quickly (NFR-PERF-04).
  row.addEventListener("toggle", () => {
    if (row.open && !body.childElementCount) body.append(...info.body().flat().filter(Boolean));
  });
  return row;
}

function stepLabel(step) {
  return step === 0 ? "Before the first model call" : `Step ${step}`;
}

function timeline(events) {
  const box = h("div", { class: "timeline" });
  let step = null;
  for (const event of events) {
    if (event.step !== step) {
      step = event.step;
      box.append(h("div", { class: "step-head" }, stepLabel(step)));
    }
    box.append(eventRow(event));
  }
  if (!events.length) box.append(h("div", { class: "side-empty" }, "No events match."));
  return box;
}

function matches(event, group, query) {
  if (EVENT_GROUPS[group] && !EVENT_GROUPS[group].includes(event.type)) return false;
  return !query || JSON.stringify(event.payload).toLowerCase().includes(query);
}

// ---------------------------------------------------------------- one trial

async function trialView(trialId) {
  const trial = await api(`/trials/${enc(trialId)}`);
  const tokens = trial.input_tokens == null ? "n/a" : String(trial.input_tokens + trial.output_tokens);
  const outcome = OUTCOME_LABEL[wellKind(trial)] || trial.outcome;
  const overridden = trial.original_outcome && trial.original_outcome !== trial.outcome;
  const facts = [
    ["Outcome", overridden ? `${outcome} (set by a reviewer; graders said ${OUTCOME_LABEL[trial.original_outcome]})` : outcome],
    ["Termination", trial.termination || "n/a"],
    ["Score", trial.score == null ? "n/a" : trial.score.toFixed(2)],
    ["Cost", usd(trial.cost_usd)],
    ["Steps", trial.steps ?? "n/a"],
    ["Tokens", tokens],
    ["Duration", trial.duration_s == null ? "n/a" : `${trial.duration_s.toFixed(1)} s`],
  ];
  const reviewable = trial.outcome === "pass" || trial.outcome === "fail";
  const graders = trial.grades.map((grade) =>
    h(
      "div",
      { class: "grader" },
      h("span", { class: `outcome ${grade.passed ? "pass" : "fail"}` }, verdictText(grade.passed)),
      " ",
      h("strong", {}, grade.grader_name),
      " ",
      h("span", { class: "class-tag" }, KIND_LABEL[grade.kind] || grade.kind),
      grade.metadata?.grader_error ? h("span", { class: "muted" }, " (the grader itself failed on this state)") : null,
      grade.kind === "judge" ? judgeDetails(grade.metadata) : null,
      pre(grade.rationale),
      grade.kind === "judge" && reviewable ? h("div", { class: "toolbar" }, queueButton(trial.trial_id, grade.grader_name, "Review this verdict")) : null,
    ),
  );
  const vetoed = trial.grades.some((grade) => grade.kind === "tamper" && !grade.passed);
  const panes = {
    timeline: () => trajectoryPane(trial),
    diff: () => diffPane(trial),
    record: () => [pre(json(trial))],
  };
  const tabs = tabSet([["timeline", "Timeline"], ["diff", "Final state diff"], ["record", "Raw record"]], panes);
  return [
    h("h1", {}, `${trial.task_id}, trial ${trial.trial_index + 1}`),
    h(
      "p",
      { class: "lede" },
      "Part of run ",
      h("a", { href: `#/runs/${enc(trial.run_id)}` }, shortId(trial.run_id)),
      trial.attempt > 1 ? `. Attempt ${trial.attempt}; earlier attempts ended in infra errors and were retried.` : ".",
    ),
    h("div", { class: "facts" }, facts.map(([label, value]) => h("span", {}, `${label} `, h("b", {}, String(value))))),
    trial.failure_tags.length
      ? h("div", { class: "tags" }, h("span", { class: "muted" }, "Failure tags"), trial.failure_tags.map((tag) => h("span", { class: "class-tag", title: `Added by ${tag.source}` }, tag.tag.replaceAll("_", " "))))
      : null,
    trial.error_code ? h("div", { class: "notice" }, h("strong", {}, trial.error_code), " ", trial.error) : null,
    vetoed
      ? h(
          "div",
          { class: "notice critical" },
          h("strong", {}, "Vetoed. "),
          "The tamper check found changes outside the allowed paths or to protected files, so the trial fails whatever the other graders said.",
        )
      : null,
    h("h2", {}, "Graders"),
    graders.length ? h("div", { class: "graders" }, graders) : h("p", { class: "muted" }, "This trial was not graded."),
    reviewable ? h("div", { class: "toolbar" }, queueButton(trial.trial_id, null, "Add to review queue")) : null,
    tabs,
  ];
}

function judgeDetails(metadata = {}) {
  const parts = [
    metadata.judge_model,
    metadata.confidence == null ? null : `confidence ${metadata.confidence.toFixed(2)}`,
    metadata.votes > 1 ? plural(metadata.votes, "vote") : null,
  ].filter(Boolean);
  return parts.length ? h("span", { class: "muted" }, ` ${parts.join(", ")}`) : null;
}

function queueButton(trialId, graderName, label) {
  const button = h("button", { type: "button" }, label);
  const result = h("span", { class: "muted", role: "status" });
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api("/review/queue", { method: "POST", body: { trial_id: trialId, grader_name: graderName } });
      result.textContent = "Added to the review queue.";
    } catch (error) {
      result.textContent = sentence(error.message);
      button.disabled = error.status === 409;
    }
  });
  return [button, result];
}

function tabSet(tabs, panes) {
  const pane = h("div", {});
  const buttons = tabs.map(([key, label]) => {
    const button = h("button", { type: "button", role: "tab", "aria-selected": "false" }, label);
    button.addEventListener("click", () => select(key, button));
    return button;
  });
  async function select(key, button) {
    for (const other of buttons) other.setAttribute("aria-selected", String(other === button));
    pane.replaceChildren(h("p", { class: "muted" }, "Loading."));
    try {
      pane.replaceChildren(...(await panes[key]()).flat().filter(Boolean));
    } catch (error) {
      pane.replaceChildren(errorState(error));
    }
  }
  select(tabs[0][0], buttons[0]);
  return [h("div", { class: "tabs", role: "tablist" }, buttons), pane];
}

async function trajectoryPane(trial) {
  const attempts = trial.attempts.map((item) => item.attempt);
  let attempt = trial.attempt;
  let events = await allEvents(trial.trial_id, attempt);
  const holder = h("div", {});
  const group = h(
    "select",
    { "aria-label": "Event types" },
    h("option", { value: "all" }, "All events"),
    h("option", { value: "model" }, "Model calls"),
    h("option", { value: "tools" }, "Tool calls"),
    h("option", { value: "messages" }, "Agent messages"),
    h("option", { value: "problems" }, "Errors and budget"),
  );
  const search = h("input", { type: "search", placeholder: "Search events (press /)", "aria-label": "Search events" });
  const count = h("span", { class: "muted" });
  const draw = () => {
    const shown = events.filter((event) => matches(event, group.value, search.value.trim().toLowerCase()));
    count.textContent = `${shown.length} of ${plural(events.length, "event")}`;
    holder.replaceChildren(timeline(shown));
  };
  group.addEventListener("change", draw);
  search.addEventListener("input", draw);
  const controls = [group, search, count];
  if (attempts.length > 1) {
    const picker = h("select", { "aria-label": "Attempt" }, attempts.map((n) => h("option", { value: n, selected: n === attempt }, `Attempt ${n}`)));
    picker.addEventListener("change", async () => {
      attempt = Number(picker.value);
      events = await allEvents(trial.trial_id, attempt);
      draw();
    });
    controls.unshift(picker);
  }
  draw();
  return [h("div", { class: "toolbar" }, controls), holder];
}

async function diffPane(trial) {
  const { diff } = await api(`/trials/${enc(trial.trial_id)}/diff`);
  const lines = diff.split("\n").filter(Boolean);
  if (!lines.length) return [h("p", { class: "muted" }, "The agent changed no files.")];
  return [
    h("p", { class: "muted" }, "Every path the trial changed in its container: A added, C changed, D deleted."),
    h("div", { class: "panel diff code" }, lines.map((line) => h("div", { class: line[0] }, line))),
  ];
}

// ---------------------------------------------------------------- compare

function pickPair(task) {
  const prefer = (trials, outcome) => trials.find((trial) => trial.outcome === outcome) || trials[0];
  if (task.delta < 0) return [prefer(task.baseline_trials, "pass"), prefer(task.candidate_trials, "fail")];
  if (task.delta > 0) return [prefer(task.baseline_trials, "fail"), prefer(task.candidate_trials, "pass")];
  return [task.baseline_trials[0], task.candidate_trials[0]];
}

async function compareView(baselineId, candidateId) {
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

// ---------------------------------------------------------------- side by side

async function sideView(leftId, rightId) {
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

// ---------------------------------------------------------------- review queue

async function reviewView() {
  keyHandler = null; // the previous item's shortcuts must not act on whatever shows next
  // Enough items that one is left after the ones skipped this visit.
  const limit = Math.min(skippedReviews.size + 1, 100);
  const { pending, items } = await api(`/review/queue?limit=${limit}`);
  const item = items.find((candidate) => !skippedReviews.has(candidate.review_id));
  if (!item) return reviewEmpty(pending);

  const score = h("input", { type: "number", id: "review-score", min: 0, max: 1, step: 0.05, inputmode: "decimal" });
  const note = h("textarea", { id: "review-note", rows: 2, maxlength: 2000 });
  const override = h("input", { type: "checkbox", id: "review-override" });
  const pass = h("button", { type: "button", class: "verdict-button pass" }, "Pass", h("kbd", {}, "p"));
  const fail = h("button", { type: "button", class: "verdict-button fail" }, "Fail", h("kbd", {}, "f"));
  const skip = h("button", { type: "button" }, "Skip", h("kbd", {}, "s"));
  const outcome = h("div", { class: "reveal", role: "status" });
  let submitted = false;
  const next = () => render(reviewView);

  async function submit(passed) {
    if (submitted) return;
    const scoreValue = score.value === "" ? null : Number(score.value);
    if (scoreValue != null && !(scoreValue >= 0 && scoreValue <= 1)) {
      outcome.replaceChildren(h("div", { class: "error-state" }, "Score must be a number from 0 to 1."));
      return;
    }
    submitted = true;
    for (const button of [pass, fail, skip]) button.disabled = true;
    try {
      const body = { passed, score: scoreValue, note: note.value.trim() || null, override: override.checked };
      const result = await api(`/review/${enc(item.review_id)}`, { method: "POST", body });
      const nextButton = h("button", { type: "button", class: "primary" }, "Next item", h("kbd", {}, "n"));
      nextButton.addEventListener("click", next);
      outcome.replaceChildren(...revealed(item, passed, result), h("div", { class: "toolbar" }, nextButton));
      nextButton.focus();
    } catch (error) {
      submitted = error.status === 409; // already reviewed elsewhere: nothing left to submit
      for (const button of [pass, fail, skip]) button.disabled = submitted;
      outcome.replaceChildren(errorState(error));
    }
  }
  function skipItem() {
    if (submitted) return;
    skippedReviews.add(item.review_id);
    next();
  }
  pass.addEventListener("click", () => submit(true));
  fail.addEventListener("click", () => submit(false));
  skip.addEventListener("click", skipItem);
  keyHandler = (key) => {
    if (key === "p") submit(true);
    else if (key === "f") submit(false);
    else if (key === "s") skipItem();
    else if (key === "n" && submitted) next();
    else return false;
    return true;
  };

  const question = item.grader_name
    ? [
        h("p", { class: "lede" }, `Answer the rubric of the judge grader ${item.grader_name}, not whether the whole trial passed. The judge's verdict stays hidden until you submit.`),
        h("h2", {}, "Rubric"),
        pre(item.rubric || "(The rubric could not be found.)"),
      ]
    : [h("p", { class: "lede" }, "Trial audit: did the agent do what the task asked? The graders' verdict stays hidden until you submit.")];
  return [
    h("h1", {}, `Review: ${item.task_id}`),
    h(
      "div",
      { class: "tags" },
      h("span", { class: "class-tag" }, SOURCE_LABEL[item.sample_source] || item.sample_source),
      h("span", { class: "muted" }, `${plural(pending, "item")} waiting, run ${shortId(item.run_id)}`),
    ),
    question,
    h("h2", {}, "Task"),
    pre(item.instructions || "(The task instructions could not be found.)"),
    h("h2", {}, "Final message from the agent"),
    pre(item.final_message || "(none)"),
    h("h2", {}, "Files changed"),
    item.changed_files.length
      ? h("div", { class: "panel diff code" }, item.changed_files.map((line) => h("div", { class: line[0] }, line)))
      : h("p", { class: "muted" }, "The agent changed no files."),
    h("h2", {}, "What the agent did"),
    pre(item.actions.length ? item.actions.join("\n") : "(no actions)"),
    h(
      "div",
      { class: "decide" },
      h(
        "div",
        { class: "fields" },
        h("label", { for: "review-score" }, "Score (optional, 0 to 1)", score),
        h("label", { for: "review-note", class: "grow" }, "Note (optional)", note),
        h("label", { for: "review-override", class: "check" }, override, "Also set the trial's outcome to my verdict"),
      ),
      h("div", { class: "toolbar" }, pass, fail, skip),
      outcome,
    ),
  ];
}

function revealed(item, passed, result) {
  const parts = [];
  if (result.judge) {
    const agreed = result.judge.passed === passed;
    const confidence = result.judge.metadata?.confidence;
    parts.push(
      h(
        "p",
        {},
        h("strong", {}, agreed ? "You agreed with the judge. " : "You disagreed with the judge. "),
        `You said ${verdictText(passed)}; the judge said ${verdictText(result.judge.passed)}`,
        confidence == null ? "." : ` with confidence ${confidence.toFixed(2)}.`,
      ),
      pre(result.judge.rationale),
    );
  } else {
    const graded = result.original_outcome || result.outcome;
    parts.push(h("p", {}, `You said ${verdictText(passed)}; the graders said ${OUTCOME_LABEL[graded] || graded}.`));
  }
  if (result.original_outcome && result.original_outcome !== result.outcome) {
    parts.push(h("p", {}, `The trial's outcome is now ${OUTCOME_LABEL[result.outcome]}, set by a reviewer.`));
  }
  parts.push(h("p", {}, h("a", { href: `#/trials/${enc(item.trial_id)}` }, "Open the trial")));
  return parts;
}

function reviewEmpty(pending) {
  if (!pending) {
    return [
      h("h1", {}, "Review"),
      h(
        "p",
        { class: "lede" },
        "The review queue is empty. Runs add trials to it at random (5% by default, set with ",
        h("code", {}, "--review-rate"),
        ") and whenever an LLM judge is unsure. You can also add a trial from its page.",
      ),
    ];
  }
  const again = h("button", { type: "button" }, "Show skipped items again");
  again.addEventListener("click", () => {
    skippedReviews.clear();
    render(reviewView);
  });
  return [h("h1", {}, "Review"), h("p", { class: "lede" }, `You skipped all ${plural(pending, "remaining item")}.`), again];
}

// ---------------------------------------------------------------- judge calibration

async function calibrationView() {
  const { judges, kappa_threshold: threshold, min_reviews: minimum } = await api("/calibration");
  const intro = h(
    "p",
    { class: "lede" },
    `Agreement and Cohen's κ count only randomly sampled reviews. A judge is calibrated when κ reaches ${threshold} over at least ${minimum} of them. Low-confidence and hand-added reviews show up among the disagreements only.`,
  );
  if (!judges.length) return [h("h1", {}, "Judge calibration"), intro, h("p", { class: "muted" }, "No judge verdicts have been reviewed yet.")];
  const status = (judge) => {
    const label = CALIBRATION_LABEL[judge.status] || judge.status;
    if (judge.status === "insufficient_data") return `${label} (${judge.random_reviews} of ${minimum})`;
    if (judge.status === "undefined") return `${label}: one side gave a single answer throughout`;
    return label;
  };
  const rows = judges.map((judge) =>
    h(
      "tr",
      {},
      h("td", {}, judge.task_id),
      h("td", {}, judge.grader_name),
      h("td", {}, h("span", { class: `status-tag ${judge.status}` }, status(judge))),
      h("td", { class: "num" }, judge.kappa == null ? "n/a" : judge.kappa.toFixed(2)),
      h("td", { class: "num" }, pct(judge.agreement)),
      h("td", { class: "num" }, String(judge.random_reviews)),
      h("td", {}, judge.judge_model || "n/a"),
    ),
  );
  return [
    h("h1", {}, "Judge calibration"),
    intro,
    table(["Task", "Grader", "Status", "κ", "Agreement", "Random reviews", "Judge model"], rows, [3, 4, 5]),
    judges.map(judgeDetail),
  ];
}

function judgeDetail(judge) {
  const c = judge.confusion;
  const matrix = table(
    ["", "Judge said Pass", "Judge said Fail"],
    [
      h("tr", {}, h("th", { scope: "row" }, "Human said Pass"), h("td", { class: "num" }, String(c.human_pass_judge_pass)), h("td", { class: "num" }, String(c.human_pass_judge_fail))),
      h("tr", {}, h("th", { scope: "row" }, "Human said Fail"), h("td", { class: "num" }, String(c.human_fail_judge_pass)), h("td", { class: "num" }, String(c.human_fail_judge_fail))),
    ],
    [1, 2],
  );
  const disagreements = judge.disagreements.map((row) =>
    h(
      "li",
      {},
      h("a", { href: `#/trials/${enc(row.trial_id)}` }, `Trial ${shortId(row.trial_id)}`),
      `: human ${verdictText(row.human_passed)}, judge ${verdictText(row.judge_passed)} `,
      h("span", { class: "muted" }, `(${(SOURCE_LABEL[row.sample_source] || row.sample_source).toLowerCase()})`),
    ),
  );
  return [
    h("h2", {}, `${judge.task_id} / ${judge.grader_name}`),
    h("div", { class: "matrix" }, matrix),
    disagreements.length ? h("ul", { class: "disagreements" }, disagreements) : h("p", { class: "muted" }, "No disagreements."),
  ];
}

// ---------------------------------------------------------------- routing and setup

const ROUTES = [
  [/^#\/runs\/?$/, () => runsView()],
  [/^#\/runs\/([^/]+)$/, (id) => runView(id)],
  [/^#\/trials\/([^/]+)$/, (id) => trialView(id)],
  [/^#\/compare\/([^/]+)\/([^/]+)$/, (a, b) => compareView(a, b)],
  [/^#\/side\/([^/]+)\/([^/]+)$/, (a, b) => sideView(a, b)],
  [/^#\/review\/?$/, () => reviewView()],
  [/^#\/calibration\/?$/, () => calibrationView()],
];

async function render(build, { keepFocus = false } = {}) {
  try {
    const content = await build();
    view.replaceChildren(...[content].flat(Infinity).filter(Boolean));
  } catch (error) {
    view.replaceChildren(errorState(error));
  }
  if (!keepFocus) {
    window.scrollTo(0, 0);
    view.focus({ preventScroll: true });
  }
}

function takeToken() {
  const match = location.hash.match(/token=([A-Za-z0-9_-]+)/);
  if (!match) return;
  token = match[1];
  remember(TOKEN_KEY, token);
  history.replaceState(null, "", "#/runs");
}

function route() {
  takeToken();
  refreshCurrent = null;
  keyHandler = null;
  tooltip.hidden = true;
  const hash = location.hash || "#/runs";
  for (const [pattern, build] of ROUTES) {
    const match = hash.match(pattern);
    if (match) {
      render(() => build(...match.slice(1).map(decodeURIComponent)));
      return;
    }
  }
  location.hash = "#/runs";
}

function applyTheme(theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  const button = document.getElementById("theme");
  button.textContent = `Theme: ${theme}`;
  button.setAttribute("aria-label", `Color theme: ${theme}`);
}

function connectEvents() {
  if (!token || typeof EventSource === "undefined") return;
  const source = new EventSource(`/events?token=${enc(token)}`);
  let seen = false;
  source.addEventListener("runs", (event) => {
    if (seen && refreshCurrent) refreshCurrent(JSON.parse(event.data));
    seen = true;
  });
}

let theme = stored(THEME_KEY) || "system";
applyTheme(THEMES.includes(theme) ? theme : "system");
document.getElementById("theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  remember(THEME_KEY, theme);
  applyTheme(theme);
});
document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
  const search = document.querySelector('input[type="search"]');
  if (event.key === "/" && search) {
    event.preventDefault();
    search.focus();
  } else if (keyHandler?.(event.key)) {
    event.preventDefault();
  }
});
window.addEventListener("hashchange", route);
takeToken();
connectEvents();
route();
