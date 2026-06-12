interface ConfidenceBandProps {
  ciLower: number;
  ciUpper: number;
  pointEstimate: number;
  isSignificant: boolean;
  label?: string;
  formatValue?: (n: number) => string;
}

export function ConfidenceBand({
  ciLower,
  ciUpper,
  pointEstimate,
  isSignificant,
  label,
  formatValue = (n) => (n * 100).toFixed(2) + '%',
}: ConfidenceBandProps) {
  // Calculate positions relative to the range
  // We want zero line visible, and CI band shown relative to it
  const absMax = Math.max(Math.abs(ciLower), Math.abs(ciUpper)) * 1.3 || 0.01;
  const rangeMin = -absMax;
  const rangeMax = absMax;
  const range = rangeMax - rangeMin;

  const toPercent = (v: number) => ((v - rangeMin) / range) * 100;

  const zeroPos = toPercent(0);
  const lowerPos = toPercent(ciLower);
  const upperPos = toPercent(ciUpper);
  const pointPos = toPercent(pointEstimate);

  const bandColor = isSignificant
    ? pointEstimate > 0 ? 'bg-emerald-400' : 'bg-red-400'
    : 'bg-slate-300';

  const pointColor = isSignificant
    ? pointEstimate > 0 ? 'bg-emerald-600' : 'bg-red-600'
    : 'bg-slate-500';

  return (
    <div className="space-y-1">
      {label && (
        <div className="flex justify-between text-sm">
          <span className="font-medium text-slate-700">{label}</span>
          <span className={isSignificant ? (pointEstimate > 0 ? 'text-emerald-600' : 'text-red-600') : 'text-slate-500'}>
            {pointEstimate > 0 ? '+' : ''}{formatValue(pointEstimate)}
          </span>
        </div>
      )}
      <div className="relative h-8 bg-slate-100 rounded-lg overflow-hidden">
        {/* Zero line */}
        <div
          className="absolute top-0 bottom-0 w-px bg-slate-400 z-10"
          style={{ left: `${zeroPos}%` }}
        />
        {/* CI band */}
        <div
          className={`absolute top-2 bottom-2 ${bandColor} rounded opacity-60`}
          style={{ left: `${lowerPos}%`, width: `${upperPos - lowerPos}%` }}
        />
        {/* Point estimate */}
        <div
          className={`absolute top-1 bottom-1 w-1 ${pointColor} rounded z-20`}
          style={{ left: `${pointPos}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-400">
        <span>{formatValue(ciLower)}</span>
        <span>0</span>
        <span>{formatValue(ciUpper)}</span>
      </div>
    </div>
  );
}
