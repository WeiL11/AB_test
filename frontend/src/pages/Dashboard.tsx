import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FlaskConical, ArrowRight } from 'lucide-react';
import { Header } from '../components/layout/Header';
import { StatusBadge } from '../components/common/StatusBadge';
import { MetricCard } from '../components/common/MetricCard';
import { useExperiments } from '../hooks/useExperiment';
import { formatDate, daysBetween } from '../utils/format';

const statusFilters = ['all', 'draft', 'running', 'completed'] as const;

export function Dashboard() {
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [page, setPage] = useState(1);
  const navigate = useNavigate();

  const status = statusFilter === 'all' ? undefined : statusFilter;
  const { data, isLoading, error } = useExperiments(page, status);

  const experiments = data?.experiments ?? [];
  const total = data?.total ?? 0;

  return (
    <div>
      <Header
        title="Dashboard"
        subtitle={`${total} experiment${total !== 1 ? 's' : ''} total`}
      />

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Total Experiments" value={total} />
        <MetricCard label="Running" value={experiments.filter(e => e.status === 'running').length} trend="up" />
        <MetricCard label="Completed" value={experiments.filter(e => e.status === 'completed').length} />
        <MetricCard label="Draft" value={experiments.filter(e => e.status === 'draft').length} />
      </div>

      {/* Status Filters */}
      <div className="flex gap-2 mb-6">
        {statusFilters.map(f => (
          <button
            key={f}
            onClick={() => { setStatusFilter(f); setPage(1); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              statusFilter === f
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Loading / Error / Empty */}
      {isLoading && (
        <div className="text-center py-12 text-slate-500">Loading experiments...</div>
      )}
      {error && (
        <div className="text-center py-12 text-red-500">
          Error loading experiments: {(error as Error).message}
        </div>
      )}
      {!isLoading && experiments.length === 0 && (
        <div className="text-center py-12">
          <FlaskConical className="w-12 h-12 text-slate-300 mx-auto mb-4" />
          <p className="text-slate-500">No experiments found.</p>
          <button
            onClick={() => navigate('/experiments/new')}
            className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700"
          >
            Create First Experiment
          </button>
        </div>
      )}

      {/* Experiment List */}
      <div className="space-y-3">
        {experiments.map(exp => (
          <div
            key={exp.id}
            onClick={() => navigate(`/experiments/${exp.id}`)}
            className="bg-white rounded-xl border border-slate-200 p-5 hover:border-indigo-300 hover:shadow-sm cursor-pointer transition-all group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h3 className="text-base font-semibold text-slate-900">{exp.name}</h3>
                <StatusBadge status={exp.status} />
              </div>
              <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-indigo-600 transition-colors" />
            </div>
            {exp.hypothesis && (
              <p className="text-sm text-slate-500 mt-2 line-clamp-1">{exp.hypothesis}</p>
            )}
            <div className="flex gap-6 mt-3 text-xs text-slate-400">
              <span>Type: <strong className="text-slate-600">{exp.analysis_type}</strong></span>
              <span>Variants: <strong className="text-slate-600">{exp.variants.length}</strong></span>
              <span>Metrics: <strong className="text-slate-600">{exp.metrics.length}</strong></span>
              <span>Created: <strong className="text-slate-600">{formatDate(exp.created_at)}</strong></span>
              {exp.started_at && (
                <span>Duration: <strong className="text-slate-600">{daysBetween(exp.started_at, exp.ended_at)} days</strong></span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {total > 20 && (
        <div className="flex justify-center gap-2 mt-6">
          <button
            disabled={page <= 1}
            onClick={() => setPage(p => p - 1)}
            className="px-3 py-1.5 rounded text-sm bg-white border border-slate-200 disabled:opacity-50"
          >
            Previous
          </button>
          <span className="px-3 py-1.5 text-sm text-slate-500">
            Page {page} of {Math.ceil(total / 20)}
          </span>
          <button
            disabled={page * 20 >= total}
            onClick={() => setPage(p => p + 1)}
            className="px-3 py-1.5 rounded text-sm bg-white border border-slate-200 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
