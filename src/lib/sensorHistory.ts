/**
 * Real sensor history: every reading that actually arrives from the collar is
 * appended here (localStorage). Nothing is generated or simulated — if the
 * collar has never sent data, the history is empty and the UI says so.
 */
export type HistoryKey = "skin" | "motion" | "temp" | "humidity" | "pressure" | "light";

export interface HistoryPoint {
  t: number; // epoch ms
  v: number;
}

const KEY = "mooomentum.sensorHistory.v1";
const MAX_PER_SENSOR = 2000;
const MIN_GAP_MS = 60_000; // at most one stored point per sensor per minute

type Store = Partial<Record<HistoryKey, HistoryPoint[]>>;

function read(): Store {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(KEY) || "{}") as Store;
  } catch {
    return {};
  }
}

function write(s: Store) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch { /* quota — ignore */ }
}

export function appendReading(sensor: HistoryKey, value: number, at = Date.now()) {
  if (!Number.isFinite(value)) return;
  const store = read();
  const list = store[sensor] ?? [];
  const last = list[list.length - 1];
  if (last && at - last.t < MIN_GAP_MS) return;
  list.push({ t: at, v: value });
  store[sensor] = list.slice(-MAX_PER_SENSOR);
  write(store);
  if (typeof window !== "undefined") window.dispatchEvent(new Event("sensor-history"));
}

export function getSeries(sensor: HistoryKey, sinceMs?: number): HistoryPoint[] {
  const list = read()[sensor] ?? [];
  return sinceMs ? list.filter((p) => p.t >= Date.now() - sinceMs) : list;
}

export function clearHistory() {
  write({});
  if (typeof window !== "undefined") window.dispatchEvent(new Event("sensor-history"));
}

export function average(points: HistoryPoint[]): number | null {
  if (!points.length) return null;
  return points.reduce((a, p) => a + p.v, 0) / points.length;
}
