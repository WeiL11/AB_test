import { api } from './client';
import type { SampleSizeRequest, SampleSizeResponse } from '../types/analysis';

export const analysisApi = {
  calculateSampleSize: (data: SampleSizeRequest) =>
    api.post<SampleSizeResponse>('/power/sample-size', data),
};
