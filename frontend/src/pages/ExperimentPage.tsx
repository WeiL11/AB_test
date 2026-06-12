import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Square } from 'lucide-react';
import { Header } from '../components/layout/Header';
import { StatusBadge } from '../components/common/StatusBadge';
import { MetricCard } from '../components/common/MetricCard';
import { ResultsSummary } from '../components/analysis/ResultsSummary';
import { ConfidenceIntervalChart } from '../components/analysis/ConfidenceIntervalChart';
import { MetricsTable } from '../components/analysis/MetricsTable';
import { useExperiment, useStartExperiment, useStopExperiment } from '../hooks/useExperiment';
import { useExperimentResults } from '../hooks/useAnalysis';
import { formatDate, daysBetween, formatNumber } from '../utils/format';

export function ExperimentPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: experiment, isLoading: expLoading } = useExperiment(id!);
  const { data: results, isLoading: resultsLoading } = useExperimentResults(id!);
  const startMutation = useStartExperiment();
  const stopMutation = useStopExperiment();

  if (expLoading) return <div className="text-center py-12 text-slate-500">Loading...</div>;
  if (!experiment) return <div className="text-center py-12 text-red-500">Experiment not found</div>;

  const totalSamples = results?.metrics.reduce((sum, m) => sum + m.sample_size_control + m.sample_size_treatment, 0) ?? 0;

  return (
    <div>
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700 mb-4"
      >
        <ArrowLeft className="w-4 h-4" /> Back to Dashboard
      </button>

      <Header
        title={experiment.name}
        subtitle={experiment.hypothesis || undefined}
        actions={
          <div className="flex items-center gap-3">
            <StatusBadge status={experiment.status} />
            {experiment.status === 'draft' && (
              <button
                onClick={() => startMutation.mutate(id!)}
                disabled={startMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 disabled:opacity-50"
              >
                <Play className="w-4 h-4" /> Start
              </button>
            )}
            {experiment.status === 'running' && (
              <button
                onClick={() => stopMutation.mutate(id!)}
                disabled={stopMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50"
              >
                <Square className="w-4 h-4" /> Stop
              </button>
            )}
          </div>
        }
      />

      {/* Experiment Info Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <MetricCard label="Analysis Type" value={experiment.analysis_type} />
        <MetricCard label="Variants" value={experiment.variants.length} />
        <MetricCard label="Metrics" value={experiment.metrics.length} />
        <MetricCard label="Created" value={formatDate(experiment.created_at)} />
        {experiment.started_at && (
          <MetricCard label="Duration" value={`${daysBetween(experiment.started_at, experiment.ended_at)} days`} />
        )}
      </div>

      {/* Variants Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 mb-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">Variants</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {experiment.variants.map(v => (
            <div key={v.id} className="border border-slate-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-slate-900">{v.name}</span>
                {v.is_control && (
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">Control</span>
                )}
              </div>
              <div className="text-sm text-slate-500">Traffic: {v.traffic_pct}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* Results Section */}
      {resultsLoading && (
        <div className="text-center py-8 text-slate-500">Loading results...</div>
      )}

      {results && results.metrics.length > 0 && (
        <div className="space-y-6">
          <ResultsSummary results={results} />

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Total Samples"
              value={formatNumber(totalSamples, 0)}
            />
            <MetricCard
              label="Significant Metrics"
              value={`${results.metrics.filter(m => m.is_significant).length} / ${results.metrics.length}`}
              trend={results.metrics.some(m => m.is_significant && m.absolute_effect > 0) ? 'up' : 'neutral'}
            />
            <MetricCard label="Recommendation" value={results.recommendation.replace('_', ' ')} />
            <MetricCard label="Analysis Type" value={results.analysis_type} />
          </div>

          <ConfidenceIntervalChart metrics={results.metrics} />
          <MetricsTable metrics={results.metrics} />
        </div>
      )}

      {results && results.metrics.length === 0 && (
        <div className="text-center py-8 text-slate-500 bg-white rounded-xl border border-slate-200">
          No results yet. {experiment.status === 'draft' ? 'Start the experiment to begin collecting data.' : 'Waiting for event data.'}
        </div>
      )}
    </div>
  );
}
