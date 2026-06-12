import { clsx } from 'clsx';

const statusStyles: Record<string, string> = {
  draft: 'bg-slate-100 text-slate-700',
  running: 'bg-emerald-100 text-emerald-700',
  paused: 'bg-amber-100 text-amber-700',
  completed: 'bg-blue-100 text-blue-700',
  killed: 'bg-red-100 text-red-700',
};

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span className={clsx(
      'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize',
      statusStyles[status] || 'bg-slate-100 text-slate-700',
      className,
    )}>
      {status}
    </span>
  );
}
