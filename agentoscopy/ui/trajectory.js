// Trajectories: fetching a trial's events and drawing them as a timeline of steps.

import { api } from "./api.js";
import { h, pre } from "./dom.js";
import { enc, json, OUTCOME_LABEL, usd } from "./format.js";

const PAGE_SIZE = 2000;
const EVENT_GROUPS = {
  all: null,
  model: ["model_request", "model_response"],
  tools: ["tool_call", "tool_result"],
  messages: ["agent_message"],
  problems: ["error", "budget_warning"],
};

export async function allEvents(trialId, attempt) {
  const events = [];
  for (;;) {
    const page = await api(`/trials/${enc(trialId)}/trajectory?attempt=${attempt}&offset=${events.length}&limit=${PAGE_SIZE}`);
    events.push(...page.events);
    if (!page.events.length || events.length >= page.total) return events;
  }
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

export function eventRow(event) {
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

export function stepLabel(step) {
  return step === 0 ? "Before the first model call" : `Step ${step}`;
}

export function timeline(events) {
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

export function matches(event, group, query) {
  if (EVENT_GROUPS[group] && !EVENT_GROUPS[group].includes(event.type)) return false;
  return !query || JSON.stringify(event.payload).toLowerCase().includes(query);
}
