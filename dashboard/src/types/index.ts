export interface PredictionInput {
  features: Record<string, number>;
}

export interface PredictionOutput {
  outbreak_probability: number;
  prediction: number;
  threshold_used: number;
  model_type: string;
}

export interface HistoryEntry {
  timestamp: string;
  proba: number;
  pred: number;
}

export interface FeatureNamesResponse {
  feature_names: string[];
}