import { clsx } from 'clsx';
import type { MetricResult } from '../../types/analysis';
import { formatPValue, formatPercent, formatNumber } from '../../utils/format';

interface Props { metrics: MetricResult[]; }

export function MetricsTable({ metrics }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-200">
        <h3 className="text-lg font-semibold text-slate-900">Detailed Results</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
              <th className="px-6 py-3">Metric</th>
              <th className="px-6 py-3">Type</th>
              <th className="px-6 py-3 text-right">Control</th>
              <th className="px-6 py-3 text-right">Treatment</th>
              <th className="px-6 py-3 text-right">Lift</th>
              <th className="px-6 py-3 text-right">CI (95%)</th>
              <th className="px-6 py-3 text-right">P-value</th>
              <th className="px-6 py-3 text-right">N (C / T)</th>
              <th className="px-6 py-3 text-center">Sig.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {metrics.map(m => (
              <tr key={m.metric_name} className="hover:bg-slate-50">
                <td className="px-6 py-4 font-medium text-slate-900">
                  {m.metric_name.replace(/_/g, ' ')}
                </td>
                <td className="px-6 py-4">
                  <span className={clsx(
                    'inline-flex px-2 py-0.5 rounded text-xs font-medium',
                    m.metric_type === 'primary' && 'bg-indigo-100 text-indigo-700',
                    m.metric_type === 'secondary' && 'bg-slate-100 text-slate-700',
                    m.metric_type === 'guardrail' && 'bg-amber-100 text-amber-700',
                  )}>
                    {m.metric_type}
                  </span>
                </td>
                <td className="px-6 py-4 text-right font-mono">{formatNumber(m.control_mean, 4)}</td>
                <td className="px-6 py-4 text-right font-mono">{formatNumber(m.treatment_mean, 4)}</td>
                <td className={clsx(
                  'px-6 py-4 text-right font-mono font-medium',
                  m.is_significant && m.relative_effect > 0 && 'text-emerald-600',
                  m.is_significant && m.relative_effect < 0 && 'text-red-600',
                )}>
                  {m.relative_effect > 0 ? '+' : ''}{formatPercent(m.relative_effect)}
                </td>
                <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
                  [{formatNumber(m.ci_lower, 4)}, {formatNumber(m.ci_upper, 4)}]
                </td>
                <td className="px-6 py-4 text-right font-mono">{formatPValue(m.p_value)}</td>
                <td className="px-6 py-4 text-right font-mono text-xs">
                  {formatNumber(m.sample_size_control, 0)} / {formatNumber(m.sample_size_treatment, 0)}
                </td>
                <td className="px-6 py-4 text-center">
                  {m.is_significant ? (
                    <span className="inline-block w-3 h-3 rounded-full bg-emerald-500" title="Significant" />
                  ) : (
                    <span className="inline-block w-3 h-3 rounded-full bg-slate-300" title="Not significant" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
