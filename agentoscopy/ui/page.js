// The main view: rendering a view into it, and the hooks the view on screen may set.

import { errorState } from "./components.js";

const view = document.getElementById("view");

// `refresh(runs)` takes live run updates; `keys(key)` takes single-key shortcuts and returns
// true when it used the key. Both are cleared whenever the route changes.
export const hooks = { refresh: null, keys: null };

export async function render(build, { keepFocus = false } = {}) {
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
