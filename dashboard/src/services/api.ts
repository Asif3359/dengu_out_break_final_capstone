import axios from 'axios';
import { PredictionInput, PredictionOutput, FeatureNamesResponse } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

export const predict = async (features: PredictionInput): Promise<PredictionOutput> => {
  const response = await api.post<PredictionOutput>('/api/v1/predict', features);
  return response.data;
};

export const getFeatureNames = async (): Promise<string[]> => {
  const response = await api.get<FeatureNamesResponse>('/api/v1/feature_names');
  return response.data.feature_names;
};