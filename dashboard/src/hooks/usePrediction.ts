import { useState } from "react";
import { predict } from "@/services/api";
import { PredictionInput, PredictionOutput, HistoryEntry } from "@/types";

export const usePrediction = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionOutput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const submitPrediction = async (features: PredictionInput) => {
    setLoading(true);
    setError(null);
    try {
      const data = await predict(features);
      setResult(data);
      setHistory((prev) => [
        {
          timestamp: new Date().toLocaleTimeString(),
          proba: data.outbreak_probability,
          pred: data.prediction,
        },
        ...prev.slice(0, 9), // keep last 10
      ]);
    } catch (err: unknown) {
      setError((err as Error).message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return { loading, result, error, history, submitPrediction, reset };
};
