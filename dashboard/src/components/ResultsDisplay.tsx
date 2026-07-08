'use client';

import { PredictionOutput } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
// import { Progress } from '@/components/ui/progress';

interface Props {
  result: PredictionOutput;
}

export default function ResultsDisplay({ result }: Props) {
  const probPercent = (result.outbreak_probability * 100).toFixed(2);
  const isOutbreak = result.prediction === 1;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex justify-between text-sm">
          <span>Outbreak Probability</span>
          <span className="font-bold">{probPercent}%</span>
        </div>
        {/* <Progress value={result.outbreak_probability * 100} className="h-2" /> */}

        <Alert variant={isOutbreak ? 'destructive' : 'default'}>
          <AlertTitle>{isOutbreak ? '⚠️ Outbreak Likely' : '✅ No Outbreak'}</AlertTitle>
          <AlertDescription>
            {isOutbreak
              ? 'The model predicts a high probability of outbreak. Take preventive measures.'
              : 'The model predicts no outbreak in the near future.'}
          </AlertDescription>
        </Alert>

        <div className="text-xs text-gray-500 flex justify-between">
          <span>Threshold: {result.threshold_used.toFixed(3)}</span>
          <span>Model: {result.model_type}</span>
        </div>
      </CardContent>
    </Card>
  );
}