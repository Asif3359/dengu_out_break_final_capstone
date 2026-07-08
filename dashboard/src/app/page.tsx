'use client';

import { usePrediction } from '@/hooks/usePrediction';
import PredictionForm from '@/components/PredictionForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import HistoryChart from '@/components/HistoryChart';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';

export default function Home() {
  const { loading, result, error, history, submitPrediction } = usePrediction();

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-2">🦟 Dengue Outbreak Predictor</h1>
        <p className="text-center text-gray-600 mb-8">
          Powered by XGBoost – SEARO region outbreak forecasting
        </p>

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PredictionForm onSubmit={submitPrediction} loading={loading} />
          <div className="space-y-6">
            {result ? <ResultsDisplay result={result} /> : (
              <div className="text-gray-500 text-center p-6 bg-white rounded-lg shadow">
                Submit features to see prediction.
              </div>
            )}
            <HistoryChart data={history} />
          </div>
        </div>
      </div>
    </div>
  );
}