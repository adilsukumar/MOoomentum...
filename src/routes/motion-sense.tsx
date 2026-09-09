import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import AppShell, { TopBar } from "@/components/AppShell";
import { SenseBanner } from "@/components/SenseBanner";
import { useLanguage } from "@/context/LanguageContext";
import { useT } from "@/context/LanguageContext";
import { usePet, displayName } from "@/context/PetContext";
import { Activity } from "lucide-react";
import { useCollar } from "@/context/CollarContext";
import { NoData, DASH } from "@/components/NoData";

export const Route = createFileRoute("/motion-sense")({ component: MotionSensePage });

// ---------- Tokens ----------
const C = {
  page: "var(--acc-pale)",
  card: "#FFFFFF",
  sumi: "var(--text-primary)",
  ink: "var(--text-primary)",
  ink2: "var(--text-secondary)",
  muted: "var(--text-secondary)",
  faint: "var(--acc-soft)",
  divider: "var(--acc-pale)",
  // Pastel blue identity (token names kept as `rose*` for minimal diff)
  rose: "var(--acc-strong)",       // primary soft sky blue
  roseDeep: "var(--acc2-strong)",   // medium ocean blue
  roseSoft: "var(--acc-strong)",   // pastel cornflower
  rosePale: "var(--acc-soft)",   // soft blue
  roseTint: "var(--acc-pale)",   // very light blue tint
  roseFill: "var(--acc-pale)",   // barely blue
  roseMid: "var(--acc-strong)",    // medium pastel blue
  roseTrack: "var(--acc-pale)",  // progress track
  indigo: "var(--acc-strong)",
  orange: "var(--acc-strong)",
  green: "var(--acc-strong)",
  greenBg: "var(--acc-pale)",
};

// ---------- Reusable ----------
function CardBox({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        background: C.card,
        borderRadius: 22,
        boxShadow: "0 2px 16px rgba(0,0,0,0.05)",
        padding: 20,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function SectionHeader({ jp, en, rightJp, rightEn }: { jp: string; en: string; rightJp?: string; rightEn?: string }) {
  const t = useT();
  return (
    <div className="flex items-center justify-between" style={{ marginBottom: 14 }}>
      <div className="flex items-center" style={{ gap: 6 }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: C.rose }} />
        <span style={{ fontSize: 11, color: C.rose, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" }}>
          {t(jp, en)}
        </span>
      </div>
      {rightJp && rightEn && (
        <span style={{ fontSize: 11, color: C.muted }}>{t(rightJp, rightEn)}</span>
      )}
    </div>
  );
}

// ---------- Page ----------
function MotionSensePage() {
  const t = useT();
  const { pet } = usePet();
  const name = displayName(pet, "Fluffy");
  const [tab, setTab] = useState<"1d" | "1w" | "1m">("1d");
  const { live } = useCollar();
  const movement = live.motion?.value ?? null;

  return (
    <AppShell
      noPadding
      renderTopBar={({ menuOpen, onMenuClick }) => (
        <TopBar showBack backTo="/home" menuOpen={menuOpen} onMenuClick={onMenuClick} />
      )}
    >
      <style>{`
        @keyframes msLive { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.35);opacity:.55} }
        @keyframes msCardIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes msPulse { 0%,100%{box-shadow:0 0 0 0 color-mix(in oklab, var(--acc-strong) 60.0%, transparent)} 50%{box-shadow:0 0 12px 2px color-mix(in oklab, var(--acc-strong) 50.0%, transparent)} }
        @keyframes msFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }
        .ms-stack > * { opacity:0; animation: msCardIn 350ms cubic-bezier(.2,.7,.2,1) forwards; }
        .ms-stack > *:nth-child(1){animation-delay:80ms}
        .ms-stack > *:nth-child(2){animation-delay:160ms}
        .ms-stack > *:nth-child(3){animation-delay:240ms}
        .ms-stack > *:nth-child(4){animation-delay:320ms}
        .ms-stack > *:nth-child(5){animation-delay:400ms}
        .ms-stack > *:nth-child(6){animation-delay:480ms}
      `}</style>

      <div style={{ background: C.page, minHeight: "100%", paddingBottom: 100 }}>
        {/* HEADER */}
        <SenseBanner
          subtitleEn="MotionSense"
          titleEn="MotionSense"
          descriptorEn="Activity tracking"
          bgGradient="linear-gradient(135deg,var(--acc-pale) 0%,var(--acc-pale) 100%)"
          subtitleColor="var(--acc-strong)"
        />


        {/* TIME TABS */}
        <div style={{ padding: "0 16px", marginTop: -36, position: "relative", zIndex: 2 }}>
          <div className="flex" style={{ background: "var(--bg-elevated)", borderRadius: 50, padding: 3, gap: 2 }}>
            {(["1d", "1w", "1m"] as const).map((v) => {
              const active = tab === v;
              return (
                <button key={v} onClick={() => setTab(v)} style={{
                  flex: 1, height: 30, borderRadius: 50,
                  background: active ? C.rose : "transparent",
                  color: active ? "#FFF" : C.muted,
                  fontSize: 13, fontWeight: active ? 600 : 500,
                  letterSpacing: "0.04em", transition: "all 200ms ease",
                }}>
                  {v.toUpperCase()}
                </button>
              );
            })}
          </div>
        </div>

        {/* CONTENT */}
        <div className="ms-stack" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: 14, position: "relative", zIndex: 2 }}>
          <HeroMotionCard movement={movement} />
          <CardBox>
            <SectionHeader jp="週間アクティビティ" en="Weekly Activity" />
            <NoData
              title={t("記録はまだありません", "No activity history yet")}
              hint={t("センサーのデータが記録されるとここに表示されます。", `Daily and weekly activity for ${name} will appear here once the collar has been recording.`)}
            />
          </CardBox>
        </div>
      </div>
    </AppShell>
  );
}

// ---------- Card 1: Live motion intensity ----------
function HeroMotionCard({ movement }: { movement: number | null }) {
  const t = useT();
  return (
    <div style={{
      background: C.card,
      borderRadius: 24,
      padding: 24,
      boxShadow: "0 12px 36px color-mix(in oklab, var(--acc-deep) 15.0%, transparent)",
    }}>
      <SectionHeader jp="現在の動き" en="Movement intensity (live)" />
      <div className="flex items-baseline" style={{ gap: 6 }}>
        <span style={{ fontSize: 40, fontWeight: 800, color: C.sumi, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
          {movement == null ? DASH : movement.toFixed(2)}
        </span>
        <span style={{ fontSize: 15, color: C.rose, fontWeight: 600 }}>m/s²</span>
      </div>
      <div className="flex items-center" style={{ gap: 6, marginTop: 10 }}>
        <Activity size={14} color={C.rose} />
        <span style={{ fontSize: 12, color: C.ink2 }}>
          {movement == null
            ? t("センサーからのデータを待っています", "Waiting for your collar to report movement")
            : t("コラーからのライブデータ", "Live from your collar")}
        </span>
      </div>
    </div>
  );
}
