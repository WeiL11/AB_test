import { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';
import { useSampleSize } from '../../hooks/useAnalysis';
import { formatNumber } from '../../utils/format';
import type { SampleSizeRequest } from '../../types/analysis';

export function PowerCalculator() {
  const [baselineRate, setBaselineRate] = useState(0.10);
  const [mde, setMde] = useState(0.02);
  const [alpha, setAlpha] = useState(0.05);
  const [power, setPower] = useState(0.80);
  const [dailyTraffic, setDailyTraffic] = useState(10000);
  const [metricType, setMetricType] = useState('binomial');

  const params: SampleSizeRequest = useMemo(() => ({
    baseline_rate: baselineRate,
    minimum_detectable_effect: mde,
    alpha,
    power,
    metric_type: metricType,
    daily_traffic: dailyTraffic,
  }), [baselineRate, mde, alpha, power, dailyTraffic, metricType]);

  const { data, isLoading, error } = useSampleSize(params);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Panel */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-5">
        <h3 className="text-lg font-semibold text-slate-900">Parameters</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Metric Type</label>
            <select
              value={metricType}
              onChange={e => setMetricType(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="binomial">Binomial (conversion rate)</option>
              <option value="continuous">Continuous (revenue, time)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Baseline Rate: {(baselineRate * 100).toFixed(1)}%
            </label>
            <input type="range" min="0.01" max="0.50" step="0.01" value={baselineRate}
              onChange={e => setBaselineRate(Number(e.target.value))}
              className="w-full accent-indigo-600" />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Minimum Detectable Effect (MDE): {(mde * 100).toFixed(1)}pp
            </label>
            <input type="range" min="0.001" max="0.10" step="0.001" value={mde}
              onChange={e => setMde(Number(e.target.value))}
              className="w-full accent-indigo-600" />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Significance Level (alpha): {alpha}
            </label>
            <input type="range" min="0.01" max="0.10" step="0.01" value={alpha}
              onChange={e => setAlpha(Number(e.target.value))}
              className="w-full accent-indigo-600" />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Statistical Power: {(power * 100).toFixed(0)}%
            </label>
            <input type="range" min="0.50" max="0.99" step="0.01" value={power}
              onChange={e => setPower(Number(e.target.value))}
              className="w-full accent-indigo-600" />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Daily Traffic
            </label>
            <input type="number" value={dailyTraffic} min={100}
              onChange={e => setDailyTraffic(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500" />
          </div>
        </div>
      </div>

      {/* Results Panel */}
      <div className="space-y-6">
        {/* Numbers */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Results</h3>
          {isLoading && <p className="text-sm text-slate-500">Calculating...</p>}
          {error && <p className="text-sm text-red-500">Error: {(error as Error).message}</p>}
          {data && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-slate-500">Sample Size (per variant)</p>
                <p className="text-2xl font-bold text-slate-900">{formatNumber(data.sample_size_per_variant, 0)}</p>
              </div>
              <div>
                <p className="text-sm text-slate-500">Total Sample Size</p>
                <p className="text-2xl font-bold text-slate-900">{formatNumber(data.total_sample_size, 0)}</p>
              </div>
              {data.estimated_days !== null && (
                <div>
                  <p className="text-sm text-slate-500">Estimated Duration</p>
                  <p className="text-2xl font-bold text-indigo-600">{data.estimated_days} days</p>
                </div>
              )}
              <div>
                <p className="text-sm text-slate-500">Power</p>
                <p className="text-2xl font-bold text-slate-900">{(data.power * 100).toFixed(0)}%</p>
              </div>
            </div>
          )}
        </div>

        {/* Power Curve */}
        {data && data.power_curve.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Power Curve</h3>
            <p className="text-sm text-slate-500 mb-4">
              Shows statistical power at different effect sizes for your sample size.
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.power_curve.map(p => ({
                effect: (p.effect_size * 100).toFixed(2),
                power: p.power * 100,
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="effect" label={{ value: 'Effect Size (%)', position: 'bottom', offset: -5 }} fontSize={12} />
                <YAxis domain={[0, 100]} label={{ value: 'Power (%)', angle: -90, position: 'insideLeft' }} fontSize={12} />
                <Tooltip formatter={(val: number) => `${val.toFixed(1)}%`} />
                <ReferenceLine y={80} stroke="#94a3b8" strokeDasharray="5 5" label="80%" />
                <Line type="monotone" dataKey="power" stroke="#6366f1" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
