import type { ReactNode } from "react";
import { Card, SectionLabel, SP } from "@/components/SensorPage";
import { NoData, DASH } from "@/components/NoData";
import { useCollar, type SensorKey } from "@/context/CollarContext";

/** Renders the live value for one sensor plus an honest empty history state.
 *  No values are ever invented — everything comes from the collar. */
export function LiveSensorBody({
  sensor,
  labelEn,
  unitLabel,
  decimals = 1,
  note,
  icon,
  showHistory = true,
}: {
  sensor: SensorKey;
  labelEn: string;
  unitLabel?: string;
  decimals?: number;
  note?: string;
  icon?: ReactNode;
  showHistory?: boolean;
}) {
  const { live } = useCollar();
  const reading = live[sensor];
  const value = reading ? reading.value.toFixed(decimals) : DASH;
  const unit = unitLabel ?? reading?.unit ?? "";

  return (
    <>
      <Card>
        <SectionLabel jp={labelEn} en={labelEn} />
        <div className="flex items-baseline" style={{ gap: 6 }}>
          {icon ? <span style={{ marginRight: 4 }}>{icon}</span> : null}
          <span
            style={{
              fontSize: 44,
              fontWeight: 800,
              color: SP.sumi,
              lineHeight: 1,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {value}
          </span>
          {unit ? <span style={{ fontSize: 16, fontWeight: 600, color: SP.rose }}>{unit}</span> : null}
        </div>
        <div style={{ fontSize: 12, color: SP.usuzumi, marginTop: 10, lineHeight: 1.6 }}>
          {reading
            ? `Live from your collar · updated ${new Date(reading.at).toLocaleTimeString()}`
            : "Waiting for this sensor to report a reading."}
        </div>
        {note ? (
          <div style={{ fontSize: 11.5, color: SP.muted, marginTop: 6, lineHeight: 1.6 }}>{note}</div>
        ) : null}
      </Card>

      {showHistory && (
        <Card>
          <SectionLabel jp="History" en="History" />
          <NoData
            title="No history recorded yet"
            hint="Trends and daily summaries will build up here as your collar keeps reporting."
          />
        </Card>
      )}
    </>
  );
}

export default LiveSensorBody;
