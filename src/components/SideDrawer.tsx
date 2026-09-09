import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "@tanstack/react-router";
import { Home, MapPin, Bot, HeartPulse, Users, FileHeart, BookOpen, Settings, ChevronRight, LogIn, LogOut, ScanSearch, Pill } from "lucide-react";
import { useT } from "@/context/LanguageContext";
import { usePet, displayName } from "@/context/PetContext";
import { useAuth } from "@/context/AuthContext";
import { useCollar } from "@/context/CollarContext";
import pawLogoAsset from "@/assets/paw-logo.png.asset.json";

type Item = {
  route: string;
  Icon: typeof Home;
  iconBg: string;
  iconColor: string;
  labelJp: string;
  labelEn: string;
  subJp: string;
  subEn: string;
};

const MAIN_ITEMS: Item[] = [
  { route: "/home", Icon: Home, iconBg: "var(--bg-card-sakura)", iconColor: "var(--accent-sakura)", labelJp: "ホーム", labelEn: "Home", subJp: "ダッシュボード", subEn: "Dashboard" },
  { route: "/map", Icon: MapPin, iconBg: "var(--acc2-pale)", iconColor: "var(--accent-sora)", labelJp: "地図", labelEn: "Map", subJp: "位置トラッカー", subEn: "Location Tracker" },
  { route: "/ai", Icon: Bot, iconBg: "var(--acc-pale)", iconColor: "var(--accent-fuji)", labelJp: "AIアシスタント", labelEn: "AI Assistant", subJp: "AIチャット", subEn: "MOooMENTUM AI" },
  { route: "/clinics", Icon: HeartPulse, iconBg: "var(--acc-pale)", iconColor: "var(--accent-sora)", labelJp: "クリニック", labelEn: "Clinics", subJp: "獣医を探す", subEn: "Find Vets" },
  { route: "/community", Icon: Users, iconBg: "var(--acc-pale)", iconColor: "var(--accent-yuzu)", labelJp: "コミュニティ", labelEn: "Community", subJp: "犬の家族", subEn: "Animal Families" },
];

const SECONDARY_ITEMS: Item[] = [
  { route: "/report", Icon: FileHeart, iconBg: "var(--acc-pale)", iconColor: "var(--accent-matcha)", labelJp: "健康レポート", labelEn: "Health Report", subJp: "詳細レポート", subEn: "Detailed Report" },
  { route: "/breeds", Icon: BookOpen, iconBg: "var(--accent-sakura-soft)", iconColor: "var(--accent-sakura)", labelJp: "犬種図鑑", labelEn: "Breed Guide", subJp: "犬種百科", subEn: "Encyclopedia" },
];

const SETTINGS_ITEM: Item = { route: "/settings", Icon: Settings, iconBg: "var(--bg-elevated)", iconColor: "var(--text-secondary)", labelJp: "設定", labelEn: "Settings", subJp: "設定", subEn: "Preferences" };

/* Vet role: clinical features only */
const VET_ITEMS: Item[] = [
  { route: "/home", Icon: Home, iconBg: "var(--acc-pale)", iconColor: "var(--accent-sakura)", labelJp: "コンソール", labelEn: "Console", subJp: "患者ダッシュボード", subEn: "Patient Dashboard" },
  { route: "/vet-consult", Icon: ScanSearch, iconBg: "var(--acc-pale)", iconColor: "var(--accent-sakura)", labelJp: "ボディマップ", labelEn: "Body Map", subJp: "タップで記録", subEn: "Tap-to-log Exam" },
  { route: "/vet-rx", Icon: Pill, iconBg: "var(--acc2-pale)", iconColor: "var(--accent-sora)", labelJp: "処方せん", labelEn: "e-Rx & Toxins", subJp: "安全性エンジン", subEn: "Safety Engine" },
];

function greeting() {
  const h = new Date().getHours();
  if (h < 11) return "Good morning!";
  if (h < 17) return "Good afternoon!";
  if (h < 21) return "Good evening!";
  return "Good night!";
}

export default function SideDrawer({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const loc = useLocation();
  const navigate = useNavigate();
  const t = useT();
  const { pet } = usePet();
  const { session, signOut, hydrated } = useAuth();
  const { connected, receiving, live, capabilities } = useCollar();
  const activeSensors = capabilities.filter((key) => live[key] != null).length;
  const collarScore = receiving && capabilities.length > 0
    ? Math.round((activeSensors / capabilities.length) * 100)
    : null;
  // Time-of-day greeting is client-only to avoid hydration mismatch.
  const [greet, setGreet] = useState("Hello!");
  useEffect(() => { setGreet(greeting()); }, []);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [isOpen, onClose]);

  const isActive = (route: string) => loc.pathname === route || loc.pathname.startsWith(route + "/");

  const handleNav = (route: string) => {
    onClose();
    setTimeout(() => navigate({ to: route }), 150);
  };

  const isVet = hydrated && session?.role === "vet";
  const mainItems = isVet ? VET_ITEMS : MAIN_ITEMS;
  const secondaryItems = isVet ? [] : SECONDARY_ITEMS;
  const BOTTOM_NAV_ROUTES = new Set(
    isVet ? ["/home", "/vet-consult", "/vet-rx"] : ["/home", "/map", "/ai", "/clinics", "/community"]
  );

  const renderItem = (it: Item, idx: number) => {
    const active = isActive(it.route);
    const inBottomNav = BOTTOM_NAV_ROUTES.has(it.route);
    const { Icon } = it;
    return (
      <button
        key={it.route}
        onClick={() => handleNav(it.route)}
        className="w-full flex items-center text-left relative"
        style={{
          height: 52,
          padding: "0 20px",
          gap: 14,
          background: active ? "linear-gradient(90deg, var(--bg-card-sakura), transparent)" : "transparent",
          borderLeft: active ? "3px solid var(--accent-sakura)" : "3px solid transparent",
          opacity: isOpen ? 1 : 0,
          transform: isOpen ? "translateX(0)" : "translateX(-20px)",
          transition: `opacity 0.3s ease ${idx * 30}ms, transform 0.3s ease ${idx * 30}ms, background 0.15s`,
        }}
      >
        <div
          className="flex items-center justify-center shrink-0"
          style={{ width: 36, height: 36, borderRadius: 10, background: active ? "var(--bg-card-sakura)" : it.iconBg }}
        >
          <Icon size={18} strokeWidth={2} style={{ color: it.iconColor }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center" style={{ gap: 6 }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: active ? "var(--accent-sakura)" : "var(--text-primary)", lineHeight: 1.2 }}>
              {t(it.labelJp, it.labelEn)}
            </div>
            {inBottomNav && (
              <span
                title={t("下のナビからアクセス可能", "Quick access from bottom nav")}
                style={{
                  fontSize: 9,
                  color: "var(--text-muted-soft)",
                  background: "var(--border-subtle)",
                  borderRadius: 6,
                  padding: "1px 5px",
                  lineHeight: 1.2,
                }}
              >
                ↓
              </span>
            )}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
            {t(it.subJp, it.subEn)}
          </div>
        </div>
        {active ? (
          <span className="pulse" style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent-sakura)" }} />
        ) : (
          <ChevronRight size={14} style={{ color: "var(--text-placeholder)" }} />
        )}
      </button>
    );
  };

  const SectionLabel = ({ jp, en }: { jp: string; en: string }) => (
    <div
      style={{
        padding: "4px 20px 6px",
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: "0.08em",
        color: "var(--text-muted-soft)",
        textTransform: "uppercase",
      }}
    >
      {t(jp, en)}
    </div>
  );

  const Divider = () => <div style={{ height: 1, background: "var(--border-subtle)", margin: "8px 20px" }} />;

  const name = displayName(pet, t("ワンちゃん", "My Pet"));

  return (
    <>
      {/* Overlay */}
      <div
        onClick={onClose}
        aria-hidden={!isOpen}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          background: "color-mix(in oklab, var(--text-primary) 42%, transparent)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          opacity: isOpen ? 1 : 0,
          pointerEvents: isOpen ? "auto" : "none",
          transition: "opacity 0.3s ease",
          zIndex: 998,
        }}
      />
      {/* Drawer */}
      <aside
        role="dialog"
        aria-label="Navigation"
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: "min(304px, 84%)",
          background: "var(--bg-drawer)",
          boxShadow: "12px 0 40px rgba(22,62,56,0.16)",
          transform: isOpen ? "translateX(0)" : "translateX(-100%)",
          transition: isOpen
            ? "transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)"
            : "transform 0.25s ease",
          zIndex: 999,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Top */}
        <div
          className="relative overflow-hidden"
          style={{
            height: 176,
            background: "var(--bg-drawer-hero)",
            padding: "20px",
          }}
        >
          <div className="jaipur-vine" style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 16, opacity: 0.3 }} aria-hidden />

          <div className="flex items-center gap-3 relative">
            <div
              className="flex items-center justify-center overflow-hidden"
              style={{
                width: 48, height: 48, borderRadius: "50%",
                background: "var(--bg-card)", border: "1px solid var(--border-card)",
                boxShadow: "0 4px 16px color-mix(in srgb, var(--accent-sakura) calc(0.2 * 100%), transparent)",
              }}
            >
              <img src={pawLogoAsset.url} alt="MOooMENTUM logo" style={{ width: 44, height: 44, objectFit: "contain" }} />
            </div>
            <div>
               <div style={{ fontSize: 21, fontWeight: 500, color: "var(--text-primary)", lineHeight: 1, fontFamily: "var(--font-display)" }}>MOooMENTUM</div>
               <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", marginTop: 5, letterSpacing: "0.1em", textTransform: "uppercase" }}>Smart Animal Care</div>
            </div>
          </div>

          <div style={{ position: "absolute", bottom: 16, left: 20, right: 20 }}>
            <div className="flex items-center gap-2">
              <div
                className="flex items-center justify-center"
                style={{ width: 32, height: 32, borderRadius: "50%", background: "#fff", border: "1.5px solid var(--accent-sakura)", fontSize: 16 }}
              >
                
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--accent-sakura)" }}>
                  {isVet ? "Veterinary Console" : t(`${name}のせかい`, `${name}'s World`)}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 1 }}>
                  {isVet && session ? `Dr. ${session.name}` : greet}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ height: 1, background: "var(--border-subtle)" }} />

        {/* Nav items (scrollable) */}
        <div className="flex-1 overflow-y-auto" style={{ padding: "12px 0" }}>
          <SectionLabel jp="メインメニュー" en={isVet ? "Clinical Tools" : "Main"} />
          {mainItems.map((it, i) => renderItem(it, i))}
          <Divider />
          {!isVet && <SectionLabel jp="その他" en="More" />}
          {secondaryItems.map((it, i) => renderItem(it, mainItems.length + i))}
          {renderItem(SETTINGS_ITEM, mainItems.length + secondaryItems.length)}
        </div>

        {/* Bottom */}
        <div style={{ marginTop: "auto" }}>
          {!isVet && (
          <div style={{ margin: 12, padding: 12, background: "var(--bg-elevated)", borderRadius: 16 }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div
                  className="flex items-center justify-center"
                  style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--bg-card)", border: "1.5px solid var(--accent-sakura)", fontSize: 14 }}
                >
                  
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1 }}>{name}</div>
                  <div className="flex items-center gap-1" style={{ marginTop: 3 }}>
                    <span className="pulse" style={{ width: 6, height: 6, borderRadius: "50%", background: connected ? "var(--accent-matcha)" : "var(--text-placeholder)" }} />
                    <span style={{ fontSize: 11, color: connected ? "var(--accent-matcha)" : "var(--text-secondary)" }}>
                      {connected ? t("首輪接続中", "Collar Connected") : t("未接続", "Collar Not Connected")}
                    </span>
                  </div>
                </div>
              </div>
              <div style={{ fontSize: 11, color: "var(--accent-sakura)", fontWeight: 600 }}>
                {collarScore == null ? "—" : `${collarScore}/100`} ✦
              </div>
            </div>
          </div>
          )}

          <button
            className="pulse-red"
            style={{
              display: "block",
              margin: "0 12px 12px",
              width: "calc(100% - 24px)",
              height: 44,
               background: "var(--accent-red)",
               color: "var(--primary-foreground)",
              fontWeight: 700,
              borderRadius: 12,
               boxShadow: "0 6px 18px color-mix(in oklab, var(--accent-red) 26%, transparent)",
              fontSize: 14,
            }}
          >
             SOS {t("緊急", "Emergency")}
          </button>

          <button
            onClick={() => {
              onClose();
              if (session) {
                signOut();
                setTimeout(() => navigate({ to: "/auth", replace: true }), 150);
              } else {
                setTimeout(() => navigate({ to: "/auth" }), 150);
              }
            }}
            className="flex items-center justify-center"
            style={{
              margin: "0 12px 10px",
              width: "calc(100% - 24px)",
              height: 42,
              gap: 8,
              background: session ? "var(--bg-elevated)" : "linear-gradient(135deg, var(--accent-sakura), var(--accent-sakura-dark))",
               color: session ? "var(--text-secondary)" : "var(--primary-foreground)",
              fontWeight: 700,
              fontSize: 13,
              borderRadius: 12,
              border: session ? "1.5px solid var(--border-subtle)" : "none",
            }}
          >
            {session ? <LogOut size={15} /> : <LogIn size={15} />}
            {session ? `Sign Out (${session.name})` : "Sign In"}
          </button>

          <div style={{ textAlign: "center", fontSize: 9, color: "var(--text-placeholder)", paddingBottom: 8 }}>
            MOooMENTUM v1.0
          </div>
        </div>
      </aside>
    </>
  );
}

export function HamburgerButton({ isOpen, onClick }: { isOpen: boolean; onClick: () => void }) {
  const lineBase: React.CSSProperties = {
    width: 18, height: 2, background: "var(--text-primary)", borderRadius: 2,
    transition: "transform 0.3s ease, opacity 0.3s ease",
    transformOrigin: "center",
  };
  return (
    <button
      onClick={onClick}
      aria-label="Open menu"
      aria-expanded={isOpen}
      className="flex items-center justify-center shrink-0"
      style={{
        width: 40, height: 40,
        background: "var(--bg-card)",
        borderRadius: 14,
        boxShadow: "0 3px 12px rgba(22,62,56,0.06)",
        border: "1px solid var(--border-subtle)",
      }}
    >
      <span className="flex flex-col" style={{ gap: 4 }}>
        <span style={{ ...lineBase, transform: isOpen ? "rotate(45deg) translate(4px, 4px)" : "none" }} />
        <span style={{ ...lineBase, opacity: isOpen ? 0 : 1, transform: isOpen ? "scaleX(0)" : "none" }} />
        <span style={{ ...lineBase, transform: isOpen ? "rotate(-45deg) translate(4px, -4px)" : "none" }} />
      </span>
    </button>
  );
}
