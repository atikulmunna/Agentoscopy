// Agentoscopy web UI: hash-routed views over the REST API, one module per group of views.
// Everything shown comes from runs and agents, so it is untrusted text: values reach the DOM
// through textContent (the `h` helper in dom.js), never through innerHTML.

import { currentToken, saveToken } from "./api.js";
import { compareView, sideView } from "./compare.js";
import { hideTooltip } from "./components.js";
import { enc } from "./format.js";
import { hooks, render } from "./page.js";
import { remember, stored } from "./prefs.js";
import { calibrationView, reviewView } from "./review.js";
import { runsView, runView } from "./runs.js";
import { trialView } from "./trial.js";

const THEME_KEY = "agentoscopy.theme";
const THEMES = ["system", "light", "dark"];
const ROUTES = [
  [/^#\/runs\/?$/, () => runsView()],
  [/^#\/runs\/([^/]+)$/, (id) => runView(id)],
  [/^#\/trials\/([^/]+)$/, (id) => trialView(id)],
  [/^#\/compare\/([^/]+)\/([^/]+)$/, (a, b) => compareView(a, b)],
  [/^#\/side\/([^/]+)\/([^/]+)$/, (a, b) => sideView(a, b)],
  [/^#\/review\/?$/, () => reviewView()],
  [/^#\/calibration\/?$/, () => calibrationView()],
];

function takeToken() {
  const match = location.hash.match(/token=([A-Za-z0-9_-]+)/);
  if (!match) return;
  saveToken(match[1]);
  history.replaceState(null, "", "#/runs");
}

function route() {
  takeToken();
  hooks.refresh = null;
  hooks.keys = null;
  hideTooltip();
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
  const token = currentToken();
  if (!token || typeof EventSource === "undefined") return;
  const source = new EventSource(`/events?token=${enc(token)}`);
  let seen = false;
  source.addEventListener("runs", (event) => {
    if (seen && hooks.refresh) hooks.refresh(JSON.parse(event.data));
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
  } else if (hooks.keys?.(event.key)) {
    event.preventDefault();
  }
});
window.addEventListener("hashchange", route);
takeToken();
connectEvents();
route();
