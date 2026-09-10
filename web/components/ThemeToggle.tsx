"use client";

import { useEffect, useState } from "react";

type ThemeChoice = "system" | "light" | "dark";

const STORAGE_KEY = "kkb-theme";

/** Applies data-theme to <html> and persists the choice. Wrapped in
 * try/catch on every localStorage call - a private window, blocked
 * site data, or a disabled storage API must not crash the toggle,
 * it should just fall back to "system" for that session. */
function applyTheme(choice: ThemeChoice) {
  console.log("[ThemeToggle] applying theme", { choice });
  const root = document.documentElement;
  if (choice === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", choice);
  }
}

function readStoredTheme(): ThemeChoice {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") {
      console.log("[ThemeToggle] read stored theme", { stored });
      return stored;
    }
  } catch (err) {
    console.warn("[ThemeToggle] localStorage read failed, defaulting to system", err);
  }
  return "system";
}

function writeStoredTheme(choice: ThemeChoice) {
  try {
    window.localStorage.setItem(STORAGE_KEY, choice);
  } catch (err) {
    console.warn("[ThemeToggle] localStorage write failed (theme won't persist)", err);
  }
}

const OPTIONS: { value: ThemeChoice; label: string; icon: string }[] = [
  { value: "light", label: "Acik", icon: "☀" },
  { value: "system", label: "Sistem", icon: "◐" },
  { value: "dark", label: "Koyu", icon: "☽" },
];

export default function ThemeToggle() {
  // Lazy initializer only ever resolves the real value on the client (guarded
  // by `mounted` below) - during SSR/the first hydration pass the rendered
  // output ignores `choice` entirely, so there's nothing to mismatch.
  const [choice, setChoice] = useState<ThemeChoice>(() =>
    typeof window === "undefined" ? "system" : readStoredTheme()
  );
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    applyTheme(choice);
    // Standard SSR-safe "mounted" gate: the render below ignores `choice`
    // entirely until this flips, so there's no cascading-render concern -
    // this is what unblocks the very first (and only) interactive render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only: applies the value already resolved by useState's lazy initializer above.
  }, []);

  function handleSelect(next: ThemeChoice) {
    console.log("[ThemeToggle] user selected", { next, previous: choice });
    setChoice(next);
    applyTheme(next);
    writeStoredTheme(next);
  }

  // Avoid a flash of the wrong active state before localStorage is read.
  if (!mounted) return <div className="h-[30px] w-[102px]" />;

  return (
    <div
      className="flex items-center gap-0.5 rounded-full border p-0.5"
      style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      role="radiogroup"
      aria-label="Tema secimi"
    >
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={choice === opt.value}
          title={opt.label}
          onClick={() => handleSelect(opt.value)}
          className="rounded-full px-2 py-1 text-xs transition-colors"
          style={{
            background: choice === opt.value ? "var(--bronze)" : "transparent",
            color: choice === opt.value ? "#fff" : "var(--text-dim)",
          }}
        >
          {opt.icon}
        </button>
      ))}
    </div>
  );
}
