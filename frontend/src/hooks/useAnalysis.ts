import { useQuery } from '@tanstack/react-query';
import { experimentsApi } from '../api/experiments';
import { analysisApi } from '../api/analysis';
import type { SampleSizeRequest } from '../types/analysis';

export function useExperimentResults(id: string) {
  return useQuery({
    queryKey: ['results', id],
    queryFn: () => experimentsApi.getResults(id),
    enabled: !!id,
    refetchInterval: 30000, // Refresh every 30s for running experiments
  });
}

export function useSampleSize(params: SampleSizeRequest | null) {
  return useQuery({
    queryKey: ['sample-size', params],
    queryFn: () => analysisApi.calculateSampleSize(params!),
    enabled: !!params,
  });
}
