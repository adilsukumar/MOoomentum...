import { createFileRoute } from "@tanstack/react-router";
import { SensorPage } from "@/components/SensorPage";
import { LiveSensorBody } from "@/components/LiveSensorBody";

export const Route = createFileRoute("/temp-sense")({ component: TempSensePage });

function TempSensePage() {
  return (
    <SensorPage
      titleEn="TempSense AI"
      subtitleEn="TempSense AI"
      descriptorEn="Ambient temperature and humidity"
      bannerGradient="linear-gradient(135deg,var(--bg-card) 0%,var(--acc-pale) 100%)"
      bannerSubtitleColor="var(--acc-strong)"
    >
      <LiveSensorBody
        sensor="temp"
        labelEn="Ambient Temperature"
        unitLabel="°C"
        note="Ambient temperature reported by the collar's SHT40 sensor; this is not body temperature."
        showHistory={false}
      />
      <LiveSensorBody
        sensor="humidity"
        labelEn="Relative Humidity"
        unitLabel="% RH"
        note="Relative humidity reported by the collar's SHT40 sensor."
      />
    </SensorPage>
  );
}
