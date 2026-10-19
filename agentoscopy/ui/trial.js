// One trial: its facts, graders, failure tags, timeline, and final state.

import { api } from "./api.js";
import { errorState, wellKind } from "./components.js";
import { h, pre } from "./dom.js";
import { enc, json, OUTCOME_LABEL, plural, sentence, shortId, usd, verdictText } from "./format.js";
import { allEvents, matches, timeline } from "./trajectory.js";

const KIND_LABEL = {
  deterministic: "Command",
  trajectory: "Trajectory",
  tamper: "Tamper check",
  judge: "LLM judge",
  custom: "Custom",
};

export async function trialView(trialId) {
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
      trial.source_trial_id
        ? [
            trial.run_mode === "fork" ? `, a fork at step ${trial.run_labels.fork_at} of trial ` : ", a replay of trial ",
            h("a", { href: `#/trials/${enc(trial.source_trial_id)}` }, shortId(trial.source_trial_id)),
            " (",
            h("a", { href: `#/side/${enc(trial.source_trial_id)}/${enc(trial.trial_id)}` }, "side by side"),
            ")",
          ]
        : null,
      trial.attempt > 1 ? `. Attempt ${trial.attempt}; earlier attempts ended in infra errors and were retried.` : ".",
    ),
    h("div", { class: "facts" }, facts.map(([label, value]) => h("span", {}, `${label} `, h("b", {}, String(value))))),
    tagSection(trial),
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

function tagSection(trial) {
  const failed = trial.outcome === "fail";
  if (!failed && !trial.failure_tags.length) return null;
  const chips = h("span", { class: "tags" });
  const status = h("span", { class: "muted", role: "status" });
  async function change(method, path, body) {
    try {
      draw((await api(path, { method, body })).failure_tags);
      status.textContent = "";
    } catch (error) {
      status.textContent = sentence(error.message);
    }
  }
  function draw(tags) {
    chips.replaceChildren(
      ...tags.map((tag) => {
        const by = tag.source === "human" ? "a reviewer" : "the harness";
        const chip = h("span", { class: "class-tag", title: tag.note ? `${tag.note} (added by ${by})` : `Added by ${by}` }, tag.tag.replaceAll("_", " "));
        if (tag.source === "human" && failed) {
          const remove = h("button", { type: "button", class: "chip-remove", "aria-label": `Remove the tag ${tag.tag}` }, "×");
          remove.addEventListener("click", () => change("DELETE", `/trials/${enc(trial.trial_id)}/tags/${enc(tag.tag)}`));
          chip.append(remove);
        }
        return chip;
      }),
    );
    if (!tags.length) chips.append(h("span", { class: "muted" }, "none yet"));
  }
  draw(trial.failure_tags);
  if (!failed) return h("div", { class: "tags" }, h("span", { class: "muted" }, "Failure tags"), chips);
  const options = h("datalist", { id: "failure-tags" });
  api("/failure-tags")
    .then(({ taxonomy }) => options.replaceChildren(...taxonomy.map((item) => h("option", { value: item.tag }, item.description))))
    .catch((error) => console.warn("Agentoscopy: no tag suggestions; free-form tags still work.", error));
  const input = h("input", { type: "text", list: "failure-tags", placeholder: "Add a tag", "aria-label": "Failure tag", maxlength: 40 });
  const add = h("button", { type: "button" }, "Add");
  const submit = () => {
    if (!input.value.trim()) return;
    change("POST", `/trials/${enc(trial.trial_id)}/tags`, { tag: input.value });
    input.value = "";
  };
  add.addEventListener("click", submit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") submit();
  });
  return h("div", { class: "tags" }, h("span", { class: "muted" }, "Failure tags"), chips, input, options, add, status);
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
