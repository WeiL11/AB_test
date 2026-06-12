import { api } from './client';
import type { Experiment, ExperimentListResponse, ExperimentCreate } from '../types/experiment';
import type { ExperimentResults } from '../types/analysis';

export const experimentsApi = {
  list: (page = 1, pageSize = 20, status?: string) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (status) params.set('status', status);
    return api.get<ExperimentListResponse>(`/experiments/?${params}`);
  },
  get: (id: string) => api.get<Experiment>(`/experiments/${id}`),
  create: (data: ExperimentCreate) => api.post<Experiment>('/experiments/', data),
  start: (id: string) => api.post<Experiment>(`/experiments/${id}/start`, {}),
  stop: (id: string) => api.post<Experiment>(`/experiments/${id}/stop`, {}),
  getResults: (id: string) => api.get<ExperimentResults>(`/experiments/${id}/results`),
};
