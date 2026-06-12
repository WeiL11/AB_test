import type { MetricResult } from '../../types/analysis';
import { ConfidenceBand } from '../common/ConfidenceBand';

interface Props { metrics: MetricResult[]; }

export function ConfidenceIntervalChart({ metrics }: Props) {
  // Group by metric type
  const primary = metrics.filter(m => m.metric_type === 'primary');
  const secondary = metrics.filter(m => m.metric_type === 'secondary');
  const guardrail = metrics.filter(m => m.metric_type === 'guardrail');

  const renderGroup = (title: string, items: MetricResult[]) => {
    if (items.length === 0) return null;
    return (
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">{title}</h4>
        {items.map(m => (
          <ConfidenceBand
            key={m.metric_name}
            label={m.metric_name.replace(/_/g, ' ')}
            ciLower={m.ci_lower}
            ciUpper={m.ci_upper}
            pointEstimate={m.absolute_effect}
            isSignificant={m.is_significant}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-6">
      <h3 className="text-lg font-semibold text-slate-900">Confidence Intervals</h3>
      <p className="text-sm text-slate-500">
        Bars show the 95% confidence interval for each metric's treatment effect.
        Green = significant positive. Red = significant negative. Gray = not significant.
      </p>
      {renderGroup('Primary Metrics', primary)}
      {renderGroup('Secondary Metrics', secondary)}
      {renderGroup('Guardrail Metrics', guardrail)}
    </div>
  );
}
