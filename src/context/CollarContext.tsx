import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { appendReading, type HistoryKey } from "@/lib/sensorHistory";

export type SensorKey = "skin" | "motion" | "temp" | "humidity" | "pressure" | "light";

export interface LiveReading {
  value: number;
  unit: string;
  /** Epoch milliseconds when the browser received the collar packet. */
  at: number;
}

type LiveMap = Record<SensorKey, LiveReading | null>;

const EMPTY_LIVE: LiveMap = {
  skin: null,
  motion: null,
  temp: null,
  humidity: null,
  pressure: null,
  light: null,
};

export type CollarState = "idle" | "connecting" | "reconnecting" | "connected";

interface CollarCtx {
  state: CollarState;
  connected: boolean;
  receiving: boolean;
  battery: number | null;
  live: LiveMap;
  capabilities: SensorKey[];
  deviceName: string | null;
  lastPacketAt: number | null;
  droppedPackets: number;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  sendCommand: (command: string) => Promise<void>;
  setLed: (enabled: boolean) => Promise<void>;
}

const Ctx = createContext<CollarCtx | null>(null);

/** Nordic UART Service implemented by the ESP32 collar firmware. */
export const UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
export const UART_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"; // app -> collar (write)
export const UART_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"; // collar -> app (notify)
const BATTERY_SERVICE = 0x180f;
const BATTERY_LEVEL = 0x2a19;
const SUPPORTED_PROTOCOL = 1;
const READING_STALE_MS = 5_000;
const MAX_RECONNECT_ATTEMPTS = 3;

const UNITS: Record<SensorKey, string> = {
  skin: "",
  motion: "m/s²",
  temp: "°C",
  humidity: "% RH",
  pressure: "kPa",
  light: "lux",
};

type BluetoothListener = (event: Event) => void;
interface BLEDevice {
  id: string;
  name?: string;
  gatt?: BLEServer;
  addEventListener: (type: string, listener: BluetoothListener) => void;
  removeEventListener: (type: string, listener: BluetoothListener) => void;
}
interface BLEServer {
  connected: boolean;
  connect: () => Promise<BLEServer>;
  disconnect: () => void;
  getPrimaryService: (service: string | number) => Promise<BLEService>;
}
interface BLEService {
  getCharacteristic: (characteristic: string | number) => Promise<BLECharacteristic>;
}
interface BLECharacteristic {
  startNotifications: () => Promise<BLECharacteristic>;
  stopNotifications?: () => Promise<BLECharacteristic>;
  readValue: () => Promise<DataView>;
  writeValue?: (value: BufferSource) => Promise<void>;
  writeValueWithoutResponse?: (value: BufferSource) => Promise<void>;
  addEventListener: (type: string, listener: BluetoothListener) => void;
  removeEventListener: (type: string, listener: BluetoothListener) => void;
}

interface ParsedPacket {
  live: Partial<LiveMap>;
  battery: number | null;
  capabilities: SensorKey[];
  protocol: number | null;
  sequence: number | null;
  got: boolean;
}

function normaliseSensorKey(raw: string): SensorKey | null {
  const key = raw.toLowerCase();
  if (key === "temp" || key === "temp_c" || key === "temperature" || key === "temperature_c") return "temp";
  if (key === "humidity" || key === "humidity_rh" || key === "rh") return "humidity";
  if (key === "motion" || key === "motion_mps2" || key === "activity_mps2") return "motion";
  if (key === "skin" || key === "pressure" || key === "light") return key;
  return null;
}

/** Parse a complete, newline-framed telemetry packet. Incomplete BLE chunks are never parsed. */
export function parseCollarPacket(text: string): ParsedPacket {
  const live: Partial<LiveMap> = {};
  const capabilities = new Set<SensorKey>();
  let battery: number | null = null;
  let protocol: number | null = null;
  let sequence: number | null = null;
  let got = false;
  const at = Date.now();

  const setValue = (rawKey: string, rawValue: unknown) => {
    if (rawKey === "battery" || rawKey === "battery_pct" || rawKey === "batt") {
      const value = Number(rawValue);
      if (Number.isFinite(value) && value >= 0 && value <= 100) battery = Math.round(value);
      return;
    }
    const key = normaliseSensorKey(rawKey);
    const value = Number(rawValue);
    if (!key || !Number.isFinite(value)) return;
    live[key] = { value, unit: UNITS[key], at };
    capabilities.add(key);
    got = true;
  };

  const trimmed = text.trim();
  if (!trimmed) return { live, battery, capabilities: [], protocol, sequence, got };

  try {
    const packet = JSON.parse(trimmed) as Record<string, unknown>;
    protocol = Number.isInteger(packet.v) ? Number(packet.v) : null;
    sequence = Number.isInteger(packet.seq) ? Number(packet.seq) : null;
    if (Array.isArray(packet.capabilities)) {
      for (const value of packet.capabilities) {
        if (typeof value !== "string") continue;
        const key = normaliseSensorKey(value);
        if (key) capabilities.add(key);
      }
    }
    for (const [key, value] of Object.entries(packet)) setValue(key.toLowerCase(), value);
  } catch {
    // Legacy complete-line support for USB-style key=value telemetry.
    for (const match of trimmed.matchAll(/([a-zA-Z_]+)\s*[:=]\s*(-?[\d.]+)/g)) {
      setValue(match[1].toLowerCase(), match[2]);
    }
  }

  return { live, battery, capabilities: [...capabilities], protocol, sequence, got };
}

function readBatteryValue(view: DataView): number | null {
  if (view.byteLength < 1) return null;
  const value = view.getUint8(0);
  return value <= 100 ? value : null;
}

export function CollarProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<CollarState>("idle");
  const [battery, setBattery] = useState<number | null>(null);
  const [live, setLive] = useState<LiveMap>(EMPTY_LIVE);
  const [capabilities, setCapabilities] = useState<SensorKey[]>([]);
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [lastPacketAt, setLastPacketAt] = useState<number | null>(null);
  const [droppedPackets, setDroppedPackets] = useState(0);
  const [receiving, setReceiving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const deviceRef = useRef<BLEDevice | null>(null);
  const txRef = useRef<BLECharacteristic | null>(null);
  const rxRef = useRef<BLECharacteristic | null>(null);
  const batteryRef = useRef<BLECharacteristic | null>(null);
  const bufferRef = useRef("");
  const decoderRef = useRef(new TextDecoder());
  const lastSequenceRef = useRef<number | null>(null);
  const manualDisconnectRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const notificationHandlerRef = useRef<BluetoothListener | null>(null);
  const batteryHandlerRef = useRef<BluetoothListener | null>(null);
  const disconnectHandlerRef = useRef<BluetoothListener | null>(null);
  const connectDeviceRef = useRef<(device: BLEDevice) => Promise<void>>(async () => undefined);

  useEffect(() => {
    for (const key of capabilities) {
      const reading = live[key];
      if (reading) appendReading(key as HistoryKey, reading.value, reading.at);
    }
  }, [capabilities, live]);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current != null) window.clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = null;
  }, []);

  const detachListeners = useCallback(() => {
    if (txRef.current && notificationHandlerRef.current) {
      txRef.current.removeEventListener("characteristicvaluechanged", notificationHandlerRef.current);
      void txRef.current.stopNotifications?.().catch(() => undefined);
    }
    if (batteryRef.current && batteryHandlerRef.current) {
      batteryRef.current.removeEventListener("characteristicvaluechanged", batteryHandlerRef.current);
      void batteryRef.current.stopNotifications?.().catch(() => undefined);
    }
    if (deviceRef.current && disconnectHandlerRef.current) {
      deviceRef.current.removeEventListener("gattserverdisconnected", disconnectHandlerRef.current);
    }
    txRef.current = null;
    rxRef.current = null;
    batteryRef.current = null;
  }, []);

  const resetReadings = useCallback(() => {
    setLive(EMPTY_LIVE);
    setCapabilities([]);
    setBattery(null);
    setReceiving(false);
    setLastPacketAt(null);
    setDroppedPackets(0);
    lastSequenceRef.current = null;
    bufferRef.current = "";
  }, []);

  const handlePacketLine = useCallback((line: string) => {
    const parsed = parseCollarPacket(line);
    if (parsed.protocol != null && parsed.protocol !== SUPPORTED_PROTOCOL) {
      setError(`Collar protocol v${parsed.protocol} is not supported by this app.`);
      return;
    }
    if (!parsed.got && parsed.battery == null) return;

    if (parsed.sequence != null) {
      const previous = lastSequenceRef.current;
      if (previous != null && parsed.sequence > previous + 1) {
        setDroppedPackets((count) => count + parsed.sequence! - previous - 1);
      }
      lastSequenceRef.current = parsed.sequence;
    }

    setLive((previous) => ({ ...previous, ...parsed.live }));
    setCapabilities((previous) => [...new Set([...previous, ...parsed.capabilities])]);
    if (parsed.battery != null) setBattery(parsed.battery);
    setLastPacketAt(Date.now());
    setReceiving(true);
    setError(null);
  }, []);

  const handleTelemetryNotification = useCallback<BluetoothListener>((event) => {
    const view = (event.target as unknown as { value?: DataView }).value;
    if (!view) return;
    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    bufferRef.current += decoderRef.current.decode(bytes, { stream: true });
    if (bufferRef.current.length > 8_192) {
      bufferRef.current = "";
      setError("The collar sent an oversized telemetry frame.");
      return;
    }
    const lines = bufferRef.current.split("\n");
    bufferRef.current = lines.pop() ?? "";
    for (const line of lines) handlePacketLine(line);
  }, [handlePacketLine]);

  const handleBatteryNotification = useCallback<BluetoothListener>((event) => {
    const view = (event.target as unknown as { value?: DataView }).value;
    if (view) setBattery(readBatteryValue(view));
  }, []);

  const connectDevice = useCallback(async (device: BLEDevice) => {
    if (!device.gatt) throw new Error("Selected device has no Bluetooth GATT server.");
    detachListeners();
    clearReconnectTimer();
    setState(reconnectAttemptsRef.current > 0 ? "reconnecting" : "connecting");

    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE);
    const tx = await service.getCharacteristic(UART_TX);
    const rx = await service.getCharacteristic(UART_RX);
    notificationHandlerRef.current = handleTelemetryNotification;
    tx.addEventListener("characteristicvaluechanged", handleTelemetryNotification);
    await tx.startNotifications();

    txRef.current = tx;
    rxRef.current = rx;
    deviceRef.current = device;
    setDeviceName(device.name ?? "MOooMENTUM Collar");

    try {
      const batteryService = await server.getPrimaryService(BATTERY_SERVICE);
      const batteryCharacteristic = await batteryService.getCharacteristic(BATTERY_LEVEL);
      batteryRef.current = batteryCharacteristic;
      setBattery(readBatteryValue(await batteryCharacteristic.readValue()));
      batteryHandlerRef.current = handleBatteryNotification;
      batteryCharacteristic.addEventListener("characteristicvaluechanged", handleBatteryNotification);
      await batteryCharacteristic.startNotifications();
    } catch {
      setBattery(null);
    }

    const onDisconnected: BluetoothListener = () => {
      detachListeners();
      setReceiving(false);
      if (manualDisconnectRef.current) {
        setState("idle");
        return;
      }
      if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        setState("idle");
        setError("The collar disconnected and could not be reconnected.");
        return;
      }
      reconnectAttemptsRef.current += 1;
      setState("reconnecting");
      reconnectTimerRef.current = window.setTimeout(() => {
        void connectDeviceRef.current(device).catch(() => onDisconnected(new Event("retry")));
      }, 1_000 * reconnectAttemptsRef.current);
    };
    disconnectHandlerRef.current = onDisconnected;
    device.addEventListener("gattserverdisconnected", onDisconnected);

    manualDisconnectRef.current = false;
    reconnectAttemptsRef.current = 0;
    setState("connected");
    setError(null);
  }, [clearReconnectTimer, detachListeners, handleBatteryNotification, handleTelemetryNotification]);
  connectDeviceRef.current = connectDevice;

  const connect = useCallback(() => {
    if (typeof window === "undefined" || !window.isSecureContext) {
      setError("Bluetooth requires HTTPS or localhost.");
      return;
    }
    const nav = navigator as Navigator & {
      bluetooth?: { requestDevice: (options: unknown) => Promise<BLEDevice> };
    };
    if (!nav.bluetooth) {
      setError("Web Bluetooth is unavailable. Use Chrome or Edge on Android, Windows, macOS, or ChromeOS.");
      return;
    }

    setError(null);
    manualDisconnectRef.current = false;
    reconnectAttemptsRef.current = 0;
    setState("connecting");
    void (async () => {
      try {
        const remembered = deviceRef.current;
        const device = remembered ?? await nav.bluetooth!.requestDevice({
          filters: [{ services: [UART_SERVICE] }],
          optionalServices: [BATTERY_SERVICE],
        });
        deviceRef.current = device;
        await connectDevice(device);
      } catch (cause) {
        setState("idle");
        const message = cause instanceof Error ? cause.message : String(cause);
        setError(/cancel|cancelled|User cancelled/i.test(message)
          ? null
          : `Couldn't connect to the collar: ${message}`);
      }
    })();
  }, [connectDevice]);

  const disconnect = useCallback(() => {
    manualDisconnectRef.current = true;
    clearReconnectTimer();
    detachListeners();
    try { deviceRef.current?.gatt?.disconnect(); } catch { /* already disconnected */ }
    setState("idle");
    resetReadings();
  }, [clearReconnectTimer, detachListeners, resetReadings]);

  const sendCommand = useCallback(async (command: string) => {
    const characteristic = rxRef.current;
    if (!characteristic || state !== "connected") throw new Error("The collar is not connected.");
    const payload = new TextEncoder().encode(command);
    if (characteristic.writeValueWithoutResponse) await characteristic.writeValueWithoutResponse(payload);
    else if (characteristic.writeValue) await characteristic.writeValue(payload);
    else throw new Error("The collar command characteristic is not writable.");
  }, [state]);

  const setLed = useCallback((enabled: boolean) => sendCommand(enabled ? "L" : "l"), [sendCommand]);

  useEffect(() => {
    if (state !== "connected" || lastPacketAt == null) return;
    const timer = window.setInterval(() => {
      if (Date.now() - lastPacketAt > READING_STALE_MS) setReceiving(false);
    }, 1_000);
    return () => window.clearInterval(timer);
  }, [lastPacketAt, state]);

  useEffect(() => () => {
    manualDisconnectRef.current = true;
    clearReconnectTimer();
    detachListeners();
    try { deviceRef.current?.gatt?.disconnect(); } catch { /* already disconnected */ }
  }, [clearReconnectTimer, detachListeners]);

  return (
    <Ctx.Provider value={{
      state,
      connected: state === "connected",
      receiving,
      battery,
      live,
      capabilities,
      deviceName,
      lastPacketAt,
      droppedPackets,
      error,
      connect,
      disconnect,
      sendCommand,
      setLed,
    }}>
      {children}
    </Ctx.Provider>
  );
}

export function useCollar(): CollarCtx {
  const value = useContext(Ctx);
  if (!value) throw new Error("useCollar must be used inside CollarProvider");
  return value;
}
