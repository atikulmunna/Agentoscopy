// The REST API, called with the token that `agentoscopy serve` printed.

import { remember, stored } from "./prefs.js";

const TOKEN_KEY = "agentoscopy.token";
let token = stored(TOKEN_KEY);

export const currentToken = () => token;

export function saveToken(value) {
  token = value;
  remember(TOKEN_KEY, value);
}

export async function api(path, { method = "GET", body: payload } = {}) {
  const headers = { Authorization: `Bearer ${token || ""}` };
  if (payload !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { method, headers, body: payload === undefined ? undefined : JSON.stringify(payload) });
  const body = await response.json().catch(() => null); // checked below: errors may come from a proxy, not the API
  if (!response.ok) {
    const error = new Error(body?.error?.message || `The server answered ${response.status}.`);
    error.code = body?.error?.code;
    error.status = response.status;
    throw error;
  }
  if (body === null) throw new Error("The server's answer was not JSON.");
  return body;
}
