import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { experimentsApi } from '../api/experiments';
import type { ExperimentCreate } from '../types/experiment';

export function useExperiments(page = 1, status?: string) {
  return useQuery({
    queryKey: ['experiments', page, status],
    queryFn: () => experimentsApi.list(page, 20, status),
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: ['experiment', id],
    queryFn: () => experimentsApi.get(id),
    enabled: !!id,
  });
}

export function useCreateExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: ExperimentCreate) => experimentsApi.create(data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useStartExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => experimentsApi.start(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useStopExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => experimentsApi.stop(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });
}
