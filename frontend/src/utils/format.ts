export function formatNumber(n: number, decimals = 2): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(decimals);
}

export function formatPercent(n: number, decimals = 2): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

export function formatPValue(p: number): string {
  if (p < 0.001) return '< 0.001';
  if (p < 0.01) return p.toFixed(3);
  return p.toFixed(2);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
  });
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

export function daysBetween(start: string, end?: string | null): number {
  const s = new Date(start);
  const e = end ? new Date(end) : new Date();
  return Math.ceil((e.getTime() - s.getTime()) / (1000 * 60 * 60 * 24));
}
