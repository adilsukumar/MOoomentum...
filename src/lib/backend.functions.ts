import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

const operationSchema = z.enum([
  "health",
  "motion.health",
  "motion.predict",
  "temperature.health",
  "temperature.environment",
  "temperature.surface",
  "vitals.health",
  "vitals.demo",
  "skin.health",
  "lights.health",
  "lights.get",
  "lights.custom",
  "lights.status",
  "animal.health",
  "animal.cow.temperature",
  "animal.cow.motion",
  "animal.buffalo.temperature",
  "animal.buffalo.mobility",
  "animal.buffalo.feeding",
  "animal.goat.pressure",
  "gps.devices",
  "gps.latest",
  "gps.status",
  "gps.history",
  "tracker.latest",
  "tracker.history",
]);

export type BackendOperation = z.infer<typeof operationSchema>;

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

type AuthKind = "none" | "api" | "gps-read" | "tracker";
type Route = {
  method: "GET" | "POST" | "PUT";
  path: string;
  auth: AuthKind;
};

const ROUTES: Record<BackendOperation, Route> = {
  health: { method: "GET", path: "/health", auth: "none" },
  "motion.health": { method: "GET", path: "/motion/health", auth: "none" },
  "motion.predict": { method: "POST", path: "/motion/predict", auth: "api" },
  "temperature.health": { method: "GET", path: "/temperature/health", auth: "none" },
  "temperature.environment": { method: "POST", path: "/temperature/assess/environment", auth: "api" },
  "temperature.surface": { method: "POST", path: "/temperature/assess/surface", auth: "api" },
  "vitals.health": { method: "GET", path: "/vitals/health", auth: "none" },
  "vitals.demo": { method: "POST", path: "/vitals/demo", auth: "api" },
  "skin.health": { method: "GET", path: "/skin/health", auth: "none" },
  "lights.health": { method: "GET", path: "/lights/health", auth: "none" },
  "lights.get": { method: "GET", path: "/lights/api/v1/lights", auth: "api" },
  "lights.custom": { method: "PUT", path: "/lights/api/v1/lights/custom", auth: "api" },
  "lights.status": { method: "PUT", path: "/lights/api/v1/lights/status", auth: "api" },
  "animal.health": { method: "GET", path: "/animal-ai/health", auth: "none" },
  "animal.cow.temperature": { method: "POST", path: "/animal-ai/cow/temperature/assess", auth: "api" },
  "animal.cow.motion": { method: "POST", path: "/animal-ai/cow/motion/screen", auth: "api" },
  "animal.buffalo.temperature": { method: "POST", path: "/animal-ai/buffalo/temperature/screen", auth: "api" },
  "animal.buffalo.mobility": { method: "POST", path: "/animal-ai/buffalo/mobility/screen", auth: "api" },
  "animal.buffalo.feeding": { method: "POST", path: "/animal-ai/buffalo/feeding/classify", auth: "api" },
  "animal.goat.pressure": { method: "POST", path: "/animal-ai/goat/pressure/screen", auth: "api" },
  "gps.devices": { method: "GET", path: "/gps/api/v1/devices", auth: "gps-read" },
  "gps.latest": { method: "GET", path: "/gps/api/v1/locations/{deviceId}/latest", auth: "gps-read" },
  "gps.status": { method: "GET", path: "/gps/api/v1/collars/{deviceId}/status", auth: "gps-read" },
  "gps.history": { method: "GET", path: "/gps/api/v1/locations/{deviceId}/history", auth: "gps-read" },
  "tracker.latest": { method: "GET", path: "/tracker/api/v1/location/latest", auth: "tracker" },
  "tracker.history": { method: "GET", path: "/tracker/api/v1/location/history", auth: "tracker" },
};

const requestSchema = z.object({
  operation: operationSchema,
  data: z.unknown().optional(),
  deviceId: z.string().min(1).max(128).regex(/^[A-Za-z0-9._:-]+$/).optional(),
  limit: z.number().int().min(1).max(1000).optional(),
  before: z.number().int().positive().optional(),
});

function backendBaseUrl(): URL {
  const raw = process.env.PAWSITIVE_BACKEND_URL;
  if (!raw) throw new Error("Missing PAWSITIVE_BACKEND_URL");
  const url = new URL(raw);
  const local = url.hostname === "localhost" || url.hostname === "127.0.0.1";
  if (url.protocol !== "https:" && !local) {
    throw new Error("PAWSITIVE_BACKEND_URL must use HTTPS");
  }
  return url;
}

function authHeaders(kind: AuthKind): HeadersInit {
  if (kind === "none") return {};

  const definitions: Record<Exclude<AuthKind, "none">, [string, string | undefined]> = {
    api: ["X-API-Key", process.env.PAWSITIVE_API_KEY],
    "gps-read": ["X-Read-API-Key", process.env.GPS_READ_API_KEY],
    tracker: ["X-API-Key", process.env.TRACKER_API_KEY],
  };
  const [header, value] = definitions[kind];
  if (!value) throw new Error(`Missing server credential for ${kind}`);
  return { [header]: value };
}

async function decodeResponse(response: Response): Promise<JsonValue> {
  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json")
    ? (await response.json()) as JsonValue
    : await response.text();
  if (!response.ok) {
    const detail = typeof body === "string" ? body : JSON.stringify(body);
    throw new Error(`Backend request failed (${response.status}): ${detail.slice(0, 500)}`);
  }
  return body;
}

/**
 * Server-only gateway to the Oracle services. The browser supplies an operation
 * name, never a URL or credential, so it cannot turn this into an open proxy.
 */
export const callBackend = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) => requestSchema.parse(input))
  .handler(async ({ data }) => {
    const route = ROUTES[data.operation];
    if (route.method === "GET" && data.data !== undefined) {
      throw new Error("GET operations do not accept a request body");
    }
    if (route.path.includes("{deviceId}") && !data.deviceId) {
      throw new Error("This operation requires deviceId");
    }

    const relativePath = route.path.replace(
      "{deviceId}",
      encodeURIComponent(data.deviceId ?? ""),
    );
    const url = new URL(relativePath, backendBaseUrl());
    if (data.limit !== undefined) url.searchParams.set("limit", String(data.limit));
    if (data.before !== undefined) url.searchParams.set("before", String(data.before));

    const headers: HeadersInit = {
      Accept: "application/json",
      ...authHeaders(route.auth),
    };
    const init: RequestInit = { method: route.method, headers };
    if (route.method !== "GET") {
      (headers as Record<string, string>)["Content-Type"] = "application/json";
      init.body = JSON.stringify(data.data ?? {});
    }

    const response = await fetch(url, {
      ...init,
      signal: AbortSignal.timeout(120_000),
    });
    return { status: response.status, data: await decodeResponse(response) };
  });

const uploadSchema = z.object({
  base64: z.string().min(1).max(14_000_000),
  mime: z.enum(["image/jpeg", "image/png", "image/webp"]),
  filename: z.string().min(1).max(128).default("skin-photo.jpg"),
});

/** Multipart SkinSense proxy; the API key remains on the server. */
export const predictSkin = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) => uploadSchema.parse(input))
  .handler(async ({ data }) => {
    const bytes = Uint8Array.from(Buffer.from(data.base64, "base64"));
    const form = new FormData();
    form.set("file", new Blob([bytes], { type: data.mime }), data.filename);
    const response = await fetch(new URL("/skin/predict", backendBaseUrl()), {
      method: "POST",
      headers: authHeaders("api"),
      body: form,
      signal: AbortSignal.timeout(120_000),
    });
    return { status: response.status, data: await decodeResponse(response) };
  });
