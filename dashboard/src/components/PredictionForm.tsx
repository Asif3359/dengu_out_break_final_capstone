"use client";

import { useState, useEffect } from "react";
import { PredictionInput } from "@/types";
import { getFeatureNames } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2 } from "lucide-react";

// ----------------------------------------------------------------
// 1. Base input interface
// ----------------------------------------------------------------
interface BaseInputs {
  country: string;          // e.g., "BGD"
  year: number;
  month: number;
  temperature_c: number;
  precipitation_mm: number;
  cases_lag_1: number;      // cases 1 month ago
  cases_lag_2: number;      // cases 2 months ago
  cases_lag_3: number;      // cases 3 months ago
}

// ----------------------------------------------------------------
// 2. Feature computation from base inputs
// ----------------------------------------------------------------
function computeFeatures(base: BaseInputs): Record<string, number> {
  const { country, year, month, temperature_c, precipitation_mm, cases_lag_1, cases_lag_2, cases_lag_3 } = base;

  // ----- Temporal features -----
  const dayofyear = new Date(year, month - 1, 1).getDate();
  const weekofyear = getWeekOfYear(year, month);
  const quarter = Math.floor((month - 1) / 3) + 1;
  const is_rainy_season = (month >= 6 && month <= 10) ? 1 : 0;
  const month_sin = Math.sin(2 * Math.PI * month / 12);
  const month_cos = Math.cos(2 * Math.PI * month / 12);
  const dayofyear_sin = Math.sin(2 * Math.PI * dayofyear / 365);
  const dayofyear_cos = Math.cos(2 * Math.PI * dayofyear / 365);

  // ----- Derived dengue features -----
  // Compute approximate longer lags and rolling stats from the three provided lags
  const cases_lag_6 = Math.round(cases_lag_3 * 0.8);
  const cases_lag_12 = Math.round(cases_lag_3 * 0.6);

  const dengue_lag_1 = cases_lag_1;
  const dengue_lag_2 = cases_lag_2;
  const dengue_lag_3 = cases_lag_3;
  const dengue_lag_6 = cases_lag_6;
  const dengue_lag_12 = cases_lag_12;
  const dengue_ma_3 = (cases_lag_1 + cases_lag_2 + cases_lag_3) / 3;
  const dengue_std_3 = Math.sqrt(
    ((cases_lag_1 - dengue_ma_3) ** 2 +
     (cases_lag_2 - dengue_ma_3) ** 2 +
     (cases_lag_3 - dengue_ma_3) ** 2) / 3
  );
  const dengue_ma_6 = (dengue_ma_3 + cases_lag_6) / 2;
  const dengue_std_6 = Math.sqrt(((dengue_ma_3 - dengue_ma_6) ** 2 + (cases_lag_6 - dengue_ma_6) ** 2) / 2);
  const dengue_ma_12 = (dengue_ma_6 + cases_lag_12) / 2;
  const dengue_std_12 = Math.sqrt(((dengue_ma_6 - dengue_ma_12) ** 2 + (cases_lag_12 - dengue_ma_12) ** 2) / 2);

  // ----- Weather lags (approximations) -----
  const temp_lag_1 = temperature_c * 0.98;
  const temp_lag_2 = temperature_c * 0.96;
  const temp_lag_3 = temperature_c * 0.94;
  const temp_lag_6 = temperature_c * 0.90;
  const temp_lag_12 = temperature_c * 0.85;
  const temp_ma_3 = (temperature_c + temp_lag_1 + temp_lag_2) / 3;
  const temp_std_3 = 0.8;
  const temp_ma_6 = (temp_ma_3 + temp_lag_3 + temp_lag_6) / 3;
  const temp_std_6 = 0.9;

  const precip_lag_1 = precipitation_mm * 0.9;
  const precip_lag_2 = precipitation_mm * 0.8;
  const precip_lag_3 = precipitation_mm * 0.7;
  const precip_lag_6 = precipitation_mm * 0.5;
  const precip_lag_12 = precipitation_mm * 0.3;
  const precip_ma_3 = (precipitation_mm + precip_lag_1 + precip_lag_2) / 3;
  const precip_std_3 = 0.5;
  const precip_ma_6 = (precip_ma_3 + precip_lag_3 + precip_lag_6) / 3;
  const precip_std_6 = 0.6;

  const temp_precip_interaction = temperature_c * precipitation_mm;

  // ----- Country encoding -----
  const countryMap: Record<string, number> = {
    BGD: 0, BTN: 1, IND: 2, IDN: 3, MDV: 4, MMR: 5, NPL: 6, LKA: 7, THA: 8, TLS: 9,
  };
  const ISO_A0_encoded = countryMap[country] ?? 0;

  // ----- Time index and year_normalized -----
  const time_index = 250;   // could be dynamic from dataset
  const year_normalized = (year - 1981) / (2025 - 1981);

  return {
    year,
    month,
    dayofyear,
    weekofyear,
    quarter,
    is_rainy_season,
    month_sin,
    month_cos,
    dayofyear_sin,
    dayofyear_cos,
    temperature_c,
    precipitation_mm,
    temp_precip_interaction,
    // dengue
    dengue_lag_1,
    dengue_lag_2,
    dengue_lag_3,
    dengue_lag_6,
    dengue_lag_12,
    dengue_ma_3,
    dengue_std_3,
    dengue_ma_6,
    dengue_std_6,
    dengue_ma_12,
    dengue_std_12,
    // weather lags
    temperature_c_lag_1: temp_lag_1,
    temperature_c_lag_2: temp_lag_2,
    temperature_c_lag_3: temp_lag_3,
    temperature_c_lag_6: temp_lag_6,
    temperature_c_lag_12: temp_lag_12,
    temperature_c_ma_3: temp_ma_3,
    temperature_c_std_3: temp_std_3,
    temperature_c_ma_6: temp_ma_6,
    temperature_c_std_6: temp_std_6,
    precipitation_mm_lag_1: precip_lag_1,
    precipitation_mm_lag_2: precip_lag_2,
    precipitation_mm_lag_3: precip_lag_3,
    precipitation_mm_lag_6: precip_lag_6,
    precipitation_mm_lag_12: precip_lag_12,
    precipitation_mm_ma_3: precip_ma_3,
    precipitation_mm_std_3: precip_std_3,
    precipitation_mm_ma_6: precip_ma_6,
    precipitation_mm_std_6: precip_std_6,
    time_index,
    year_normalized,
    ISO_A0_encoded,
  };
}

// Helper: get week of year (simplified)
function getWeekOfYear(year: number, month: number): number {
  const date = new Date(year, month - 1, 1);
  const startOfYear = new Date(year, 0, 1);
  const diff = (date.getTime() - startOfYear.getTime()) / (7 * 24 * 60 * 60 * 1000);
  return Math.ceil(diff) + 1;
}

// ----------------------------------------------------------------
// 3. Component
// ----------------------------------------------------------------
interface Props {
  onSubmit: (features: PredictionInput) => void;
  loading: boolean;
}

export default function PredictionForm({ onSubmit, loading }: Props) {
  const [base, setBase] = useState<BaseInputs>({
    country: "BGD",
    year: 2024,
    month: 6,
    temperature_c: 28.5,
    precipitation_mm: 5.2,
    cases_lag_1: 25,
    cases_lag_2: 30,
    cases_lag_3: 20,
  });

  const [allFeatures, setAllFeatures] = useState<Record<string, number>>({});
  const [featureNames, setFeatureNames] = useState<string[]>([]);

  useEffect(() => {
    getFeatureNames().then(setFeatureNames).catch(console.error);
  }, []);

  const handleChange = (key: keyof BaseInputs, value: string) => {
    setBase((prev) => ({ ...prev, [key]: value === "" ? 0 : parseFloat(value) }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const features = computeFeatures(base);
    setAllFeatures(features);
    onSubmit({ features });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Input Features</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Country */}
          <div className="space-y-1">
            <Label>Country</Label>
            <Select
              value={base.country}
              onValueChange={(val) => setBase({ ...base, country: val || "" })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select country" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="BGD">Bangladesh</SelectItem>
                <SelectItem value="IND">India</SelectItem>
                <SelectItem value="IDN">Indonesia</SelectItem>
                <SelectItem value="THA">Thailand</SelectItem>
                <SelectItem value="LKA">Sri Lanka</SelectItem>
                <SelectItem value="MMR">Myanmar</SelectItem>
                <SelectItem value="NPL">Nepal</SelectItem>
                <SelectItem value="MDV">Maldives</SelectItem>
                <SelectItem value="TLS">Timor-Leste</SelectItem>
                <SelectItem value="BTN">Bhutan</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Date */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <Label>Year</Label>
              <Input
                type="number"
                value={base.year}
                onChange={(e) => handleChange("year", e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label>Month</Label>
              <Input
                type="number"
                min={1}
                max={12}
                value={base.month}
                onChange={(e) => handleChange("month", e.target.value)}
              />
            </div>
          </div>

          {/* Weather */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <Label>Temperature (°C)</Label>
              <Input
                type="number"
                step="0.1"
                value={base.temperature_c}
                onChange={(e) => handleChange("temperature_c", e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label>Precipitation (mm/day)</Label>
              <Input
                type="number"
                step="0.1"
                value={base.precipitation_mm}
                onChange={(e) => handleChange("precipitation_mm", e.target.value)}
              />
            </div>
          </div>

          {/* Cases (lags) */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Recent Dengue Cases</Label>
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1">
                <Label className="text-xs">1 month ago</Label>
                <Input
                  type="number"
                  value={base.cases_lag_1}
                  onChange={(e) => handleChange("cases_lag_1", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">2 months ago</Label>
                <Input
                  type="number"
                  value={base.cases_lag_2}
                  onChange={(e) => handleChange("cases_lag_2", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">3 months ago</Label>
                <Input
                  type="number"
                  value={base.cases_lag_3}
                  onChange={(e) => handleChange("cases_lag_3", e.target.value)}
                />
              </div>
            </div>
          </div>

          {/* Preview of computed features (optional) */}
          <details className="mt-2">
            <summary className="text-sm text-blue-600 cursor-pointer">
              Show all 45 computed features
            </summary>
            <div className="mt-2 max-h-60 overflow-y-auto p-2 border rounded text-xs">
              {Object.entries(allFeatures).map(([key, value]) => (
                <div key={key} className="flex justify-between border-b py-1">
                  <span>{key}</span>
                  <span>{value.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </details>

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Predicting...
              </>
            ) : (
              "Predict Outbreak"
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}