export interface MetricResult {
  metric_name: string;
  metric_type: 'primary' | 'secondary' | 'guardrail';
  control_mean: number;
  treatment_mean: number;
  absolute_effect: number;
  relative_effect: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number;
  is_significant: boolean;
  sample_size_control: number;
  sample_size_treatment: number;
}

export interface ExperimentResults {
  experiment_id: string;
  experiment_name: string;
  status: string;
  analysis_type: string;
  metrics: MetricResult[];
  recommendation: 'ship' | 'dont_ship' | 'keep_running' | 'inconclusive';
  computed_at: string;
}

export interface PowerCurvePoint {
  effect_size: number;
  power: number;
}

export interface SampleSizeRequest {
  baseline_rate: number;
  minimum_detectable_effect: number;
  alpha?: number;
  power?: number;
  metric_type?: string;
  variance?: number;
  daily_traffic?: number;
  n_variants?: number;
}

export interface SampleSizeResponse {
  sample_size_per_variant: number;
  total_sample_size: number;
  estimated_days: number | null;
  power: number;
  alpha: number;
  minimum_detectable_effect: number;
  baseline_rate: number;
  power_curve: PowerCurvePoint[];
}
