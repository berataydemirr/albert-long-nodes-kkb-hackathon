interface StatTileProps {
  label: string;
  value: string;
  hint?: string;
  accent?: "bronze" | "silver" | "gold";
}

export default function StatTile({ label, value, hint, accent = "bronze" }: StatTileProps) {
  return (
    <div className="card flex flex-col gap-1" style={{ borderTop: `3px solid var(--${accent})` }}>
      <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
        {label}
      </span>
      <span className="mono text-2xl font-semibold" style={{ color: `var(--${accent})` }}>
        {value}
      </span>
      {hint && (
        <span className="text-xs" style={{ color: "var(--text-dim)" }}>
          {hint}
        </span>
      )}
    </div>
  );
}
