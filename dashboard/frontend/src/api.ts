// The frontend only talks to the dashboard backend (port 8001).
// The backend passes forecast requests on to the model server (port 8000).
import type { State, States } from "./grid";

const API = import.meta.env.VITE_API_URL || "http://localhost:8001";

export type SampleInfo = {
  id: string;
  label: string;
  kind: string;
  last_input_day: string;
  doy_sincos: number[];
};

export type Sample = {
  id: string;
  label: string;
  init_states: States; // [3, 6, 32, 36]
  doy_sincos: number[];
  truth: States; // [10, 6, 32, 36]
  truth_days: string[];
};

export type GridInfo = {
  lat: number[];
  lon: number[];
  land_mask: number[][];
};

export type SimulateResult = {
  rollout: States; // [steps, 6, 32, 36]
  inference_ms: number;
};

async function getJson<T>(path: string): Promise<T> {
  const reply = await fetch(API + path);
  if (!reply.ok) {
    throw new Error(`GET ${path} failed: ${reply.status} ${await reply.text()}`);
  }
  return reply.json();
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const reply = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!reply.ok) {
    throw new Error(`POST ${path} failed: ${reply.status} ${await reply.text()}`);
  }
  return reply.json();
}

export function fetchSamples(): Promise<SampleInfo[]> {
  return getJson("/samples");
}

export function fetchSample(id: string): Promise<Sample> {
  return getJson("/samples/" + encodeURIComponent(id));
}

export function fetchGrid(): Promise<GridInfo> {
  return getJson("/grid");
}

export function fetchHealth(): Promise<{ status: string; model_name: string }> {
  return postJson("/health", {});
}

// Every forecast shown in the dashboard comes from this live call.
export function simulate(sample: Sample, steps: number, perturbation: State | null): Promise<SimulateResult> {
  return postJson("/simulate", {
    init_states: sample.init_states,
    doy_sincos: sample.doy_sincos,
    steps: steps,
    perturbation: perturbation,
  });
}
