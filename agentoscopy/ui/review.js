// Human review (WF-09): the review queue, and how well judges agree with reviewers.

import { api } from "./api.js";
import { errorState, table } from "./components.js";
import { h, pre } from "./dom.js";
import { enc, OUTCOME_LABEL, pct, plural, shortId, verdictText } from "./format.js";
import { hooks, render } from "./page.js";

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
const skippedReviews = new Set(); // review items skipped during this visit

export async function reviewView() {
  hooks.keys = null; // the previous item's shortcuts must not act on whatever shows next
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
  hooks.keys = (key) => {
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

export async function calibrationView() {
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
