import { CheckCircle, XCircle, Clock, HelpCircle } from 'lucide-react';
import type { ExperimentResults } from '../../types/analysis';

const recommendationConfig = {
  ship: { icon: CheckCircle, label: 'Ship It', color: 'text-emerald-600', bg: 'bg-emerald-50', border: 'border-emerald-200', desc: 'Primary metrics show significant positive impact.' },
  dont_ship: { icon: XCircle, label: "Don't Ship", color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200', desc: 'Results are negative or guardrails are violated.' },
  keep_running: { icon: Clock, label: 'Keep Running', color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-200', desc: 'Not enough data to make a decision yet.' },
  inconclusive: { icon: HelpCircle, label: 'Inconclusive', color: 'text-slate-600', bg: 'bg-slate-50', border: 'border-slate-200', desc: 'Experiment completed without significant results.' },
};

interface Props { results: ExperimentResults; }

export function ResultsSummary({ results }: Props) {
  const config = recommendationConfig[results.recommendation];
  const Icon = config.icon;

  return (
    <div className={`rounded-xl border ${config.border} ${config.bg} p-6`}>
      <div className="flex items-center gap-3">
        <Icon className={`w-8 h-8 ${config.color}`} />
        <div>
          <h3 className={`text-lg font-bold ${config.color}`}>{config.label}</h3>
          <p className="text-sm text-slate-600">{config.desc}</p>
        </div>
      </div>
      <div className="mt-4 flex gap-6 text-sm text-slate-500">
        <span>Analysis: <strong className="text-slate-700">{results.analysis_type}</strong></span>
        <span>Metrics: <strong className="text-slate-700">{results.metrics.length}</strong></span>
        <span>Computed: <strong className="text-slate-700">{new Date(results.computed_at).toLocaleString()}</strong></span>
      </div>
    </div>
  );
}
