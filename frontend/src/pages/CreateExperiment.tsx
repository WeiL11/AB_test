import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PlusCircle, Trash2 } from 'lucide-react';
import { Header } from '../components/layout/Header';
import { useCreateExperiment } from '../hooks/useExperiment';
import type { VariantCreate, MetricCreate } from '../types/experiment';

export function CreateExperiment() {
  const navigate = useNavigate();
  const createMutation = useCreateExperiment();

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [hypothesis, setHypothesis] = useState('');
  const [analysisType, setAnalysisType] = useState('frequentist');

  const [variants, setVariants] = useState<VariantCreate[]>([
    { name: 'control', is_control: true, traffic_pct: 50 },
    { name: 'variant_a', is_control: false, traffic_pct: 50 },
  ]);

  const [metrics, setMetrics] = useState<MetricCreate[]>([
    { name: 'conversion_rate', metric_type: 'primary', data_type: 'binomial' },
  ]);

  const addVariant = () => {
    const n = variants.length;
    setVariants([...variants, { name: `variant_${String.fromCharCode(97 + n)}`, is_control: false, traffic_pct: 0 }]);
  };

  const removeVariant = (i: number) => {
    if (variants.length <= 2) return;
    setVariants(variants.filter((_, idx) => idx !== i));
  };

  const updateVariant = (i: number, field: keyof VariantCreate, value: string | number | boolean) => {
    const updated = [...variants];
    (updated[i] as any)[field] = value;
    setVariants(updated);
  };

  const addMetric = () => {
    setMetrics([...metrics, { name: '', metric_type: 'secondary', data_type: 'continuous' }]);
  };

  const removeMetric = (i: number) => {
    if (metrics.length <= 1) return;
    setMetrics(metrics.filter((_, idx) => idx !== i));
  };

  const updateMetric = (i: number, field: keyof MetricCreate, value: string | number | boolean) => {
    const updated = [...metrics];
    (updated[i] as any)[field] = value;
    setMetrics(updated);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const exp = await createMutation.mutateAsync({
        name,
        description: description || undefined,
        hypothesis: hypothesis || undefined,
        analysis_type: analysisType,
        variants,
        metrics,
      });
      navigate(`/experiments/${exp.id}`);
    } catch (err) {
      // Error shown via mutation state
    }
  };

  const inputClass = 'w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500';

  return (
    <div>
      <Header title="Create Experiment" subtitle="Define your hypothesis, variants, and metrics" />

      <form onSubmit={handleSubmit} className="space-y-6 max-w-3xl">
        {/* Basic Info */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <h3 className="text-lg font-semibold text-slate-900">Basic Information</h3>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Experiment Name *</label>
            <input type="text" required value={name} onChange={e => setName(e.target.value)} className={inputClass} placeholder="e.g., Checkout Button Color Test" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Hypothesis</label>
            <textarea value={hypothesis} onChange={e => setHypothesis(e.target.value)} className={inputClass} rows={2} placeholder="e.g., Changing the button to blue will increase conversion by 2%" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Description</label>
            <textarea value={description} onChange={e => setDescription(e.target.value)} className={inputClass} rows={2} />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Analysis Type</label>
            <select value={analysisType} onChange={e => setAnalysisType(e.target.value)} className={inputClass}>
              <option value="frequentist">Frequentist</option>
              <option value="bayesian">Bayesian</option>
              <option value="sequential">Sequential</option>
              <option value="bandit">Bandit</option>
            </select>
          </div>
        </div>

        {/* Variants */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-900">Variants</h3>
            <button type="button" onClick={addVariant} className="flex items-center gap-1 text-sm text-indigo-600 hover:text-indigo-700">
              <PlusCircle className="w-4 h-4" /> Add Variant
            </button>
          </div>
          {variants.map((v, i) => (
            <div key={i} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
              <input type="text" value={v.name} onChange={e => updateVariant(i, 'name', e.target.value)} className="flex-1 rounded border border-slate-300 px-2 py-1.5 text-sm" placeholder="Variant name" />
              <label className="flex items-center gap-1 text-sm text-slate-600">
                <input type="checkbox" checked={v.is_control} onChange={e => updateVariant(i, 'is_control', e.target.checked)} className="accent-indigo-600" />
                Control
              </label>
              <div className="flex items-center gap-1">
                <input type="number" value={v.traffic_pct} onChange={e => updateVariant(i, 'traffic_pct', Number(e.target.value))} className="w-20 rounded border border-slate-300 px-2 py-1.5 text-sm text-right" />
                <span className="text-sm text-slate-500">%</span>
              </div>
              <button type="button" onClick={() => removeVariant(i)} disabled={variants.length <= 2} className="text-slate-400 hover:text-red-500 disabled:opacity-30">
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>

        {/* Metrics */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-900">Metrics</h3>
            <button type="button" onClick={addMetric} className="flex items-center gap-1 text-sm text-indigo-600 hover:text-indigo-700">
              <PlusCircle className="w-4 h-4" /> Add Metric
            </button>
          </div>
          {metrics.map((m, i) => (
            <div key={i} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
              <input type="text" value={m.name} onChange={e => updateMetric(i, 'name', e.target.value)} className="flex-1 rounded border border-slate-300 px-2 py-1.5 text-sm" placeholder="Metric name" />
              <select value={m.metric_type} onChange={e => updateMetric(i, 'metric_type', e.target.value)} className="rounded border border-slate-300 px-2 py-1.5 text-sm">
                <option value="primary">Primary</option>
                <option value="secondary">Secondary</option>
                <option value="guardrail">Guardrail</option>
              </select>
              <select value={m.data_type} onChange={e => updateMetric(i, 'data_type', e.target.value)} className="rounded border border-slate-300 px-2 py-1.5 text-sm">
                <option value="binomial">Binomial</option>
                <option value="continuous">Continuous</option>
                <option value="count">Count</option>
                <option value="ratio">Ratio</option>
              </select>
              <button type="button" onClick={() => removeMetric(i)} disabled={metrics.length <= 1} className="text-slate-400 hover:text-red-500 disabled:opacity-30">
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>

        {/* Submit */}
        {createMutation.isError && (
          <p className="text-sm text-red-500">Error: {(createMutation.error as Error).message}</p>
        )}
        <div className="flex gap-3">
          <button type="submit" disabled={createMutation.isPending || !name} className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50">
            {createMutation.isPending ? 'Creating...' : 'Create Experiment'}
          </button>
          <button type="button" onClick={() => navigate('/')} className="px-6 py-2.5 bg-white text-slate-700 rounded-lg text-sm font-medium border border-slate-200 hover:bg-slate-50">
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
