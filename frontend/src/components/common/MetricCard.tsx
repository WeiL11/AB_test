import { clsx } from 'clsx';

interface MetricCardProps {
  label: string;
  value: string | number;
  subvalue?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({ label, value, subvalue, trend, className }: MetricCardProps) {
  return (
    <div className={clsx('bg-white rounded-xl border border-slate-200 p-5', className)}>
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className={clsx(
        'text-2xl font-bold mt-1',
        trend === 'up' && 'text-emerald-600',
        trend === 'down' && 'text-red-600',
        (!trend || trend === 'neutral') && 'text-slate-900',
      )}>
        {value}
      </p>
      {subvalue && <p className="text-xs text-slate-400 mt-1">{subvalue}</p>}
    </div>
  );
}
