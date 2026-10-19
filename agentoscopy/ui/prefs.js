// Settings remembered in the browser. Storage can be unavailable (a private window, blocked
// site data); the page still works then, and forgets the settings when it closes.

export function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.warn(`Agentoscopy: cannot read ${key} from browser storage.`, error);
    return null;
  }
}

export function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    console.warn(`Agentoscopy: cannot save ${key} in browser storage; it lasts for this visit only.`, error);
  }
}
