export interface Variant {
  id: string;
  name: string;
  is_control: boolean;
  traffic_pct: number;
  description: string | null;
}

export interface Metric {
  id: string;
  name: string;
  metric_type: 'primary' | 'secondary' | 'guardrail';
  data_type: 'binomial' | 'continuous' | 'count' | 'ratio';
  minimum_detectable_effect: number | null;
  cuped_enabled: boolean;
}

export interface Experiment {
  id: string;
  name: string;
  description: string | null;
  hypothesis: string | null;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'killed';
  analysis_type: 'frequentist' | 'bayesian' | 'sequential' | 'bandit';
  allocation_pct: number;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  variants: Variant[];
  metrics: Metric[];
}

export interface ExperimentListResponse {
  experiments: Experiment[];
  total: number;
  page: number;
  page_size: number;
}

export interface VariantCreate {
  name: string;
  is_control: boolean;
  traffic_pct: number;
  description?: string;
}

export interface MetricCreate {
  name: string;
  metric_type: 'primary' | 'secondary' | 'guardrail';
  data_type: 'binomial' | 'continuous' | 'count' | 'ratio';
  minimum_detectable_effect?: number;
  cuped_enabled?: boolean;
}

export interface ExperimentCreate {
  name: string;
  description?: string;
  hypothesis?: string;
  analysis_type?: string;
  allocation_pct?: number;
  variants: VariantCreate[];
  metrics: MetricCreate[];
}
