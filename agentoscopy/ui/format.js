// Values formatted for display, and small helpers for the data views receive.

export const OUTCOME_LABEL = {
  pass: "Pass",
  fail: "Fail",
  infra_error: "Infra error",
  skipped: "Skipped",
  cancelled: "Cancelled",
  pending: "Queued",
  running: "Running",
};

const sign = (v) => (v < 0 ? "-" : "+");

export const enc = encodeURIComponent;
export const pct = (v) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);
export const points = (v) => (v == null ? "n/a" : `${sign(v)}${Math.abs(v * 100).toFixed(1)} points`);
export const usd = (v, digits = 4) => (v == null ? "n/a" : `$${v.toFixed(digits)}`);
export const signed = (v, digits, prefix = "") => (v == null ? "n/a" : `${sign(v)}${prefix}${Math.abs(v).toFixed(digits)}`);
export const shortId = (id) => String(id).slice(0, 8);
export const when = (iso) => (iso ? new Date(iso).toLocaleString() : "");
export const json = (value) => JSON.stringify(value, null, 2);
export const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
export const verdictText = (passed) => (passed ? "Pass" : "Fail");
export const sentence = (text) => `${text[0].toUpperCase()}${text.slice(1)}${/[.!?]$/.test(text) ? "" : "."}`;

export function groupBy(items, key) {
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item[key])) groups.set(item[key], []);
    groups.get(item[key]).push(item);
  }
  return groups;
}
