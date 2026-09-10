/** Shared number formatting so every panel renders numbers the same way
 * (tr-TR: comma decimal, period thousands-grouping) instead of mixing
 * Intl.NumberFormat('tr-TR', ...) in some places and raw .toFixed() (which
 * always uses a period, regardless of locale) in others. */

export function formatNumber(value: number, maxDecimals = 1): string {
  return new Intl.NumberFormat("tr-TR", { maximumFractionDigits: maxDecimals }).format(value);
}

export function formatSigned(value: number, maxDecimals = 1): string {
  const formatted = formatNumber(Math.abs(value), maxDecimals);
  return value >= 0 ? `+${formatted}` : `-${formatted}`;
}

export function formatPercent(value: number, maxDecimals = 1): string {
  return `${formatSigned(value, maxDecimals)}%`;
}
