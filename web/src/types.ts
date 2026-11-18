export interface Municipality {
  geocode: number;
  name: string;
  uf: string;
  region: string;
  population: number;
  normName?: string;
  normSearch?: string;
}

export interface MapMeta {
  weeks: number[];
  geocodes: number[];
  horizons: number[];
  inc_scale: number;
  res_scale?: number;
  res_zero?: number;
  latest_week: number;
}

export interface FanQuantiles {
  "q0.05": (number | null)[];
  "q0.25": (number | null)[];
  "q0.5": (number | null)[];
  "q0.75": (number | null)[];
  "q0.95": (number | null)[];
}

export interface FanData {
  weeks: number[];
  obs: (number | null)[];
  h1?: FanQuantiles;
  h2?: FanQuantiles;
  h4?: FanQuantiles;
  h8?: FanQuantiles;
  [key: string]: FanQuantiles | number[] | (number | null)[] | undefined;
}

export interface BenchmarkRow {
  model: string;
  label: string;
  wis: Record<string, number | null>;
  rel_wis_8: number | null;
  cov90: number | null;
  pr_auc_8: number | null;
}

export interface BenchmarksData {
  rows: BenchmarkRow[];
  reference: string;
  source: string;
}

export interface StratifiedRecord {
  model: string;
  label: string;
  stratum: string;
  horizon: number;
  wis: number | null;
  wis_log: number | null;
  rel_wis: number | null;
  rel_wis_log: number | null;
  mae: number | null;
  cov90: number | null;
}

export interface UfSkillItem {
  uf_code: number;
  uf: string;
  region: string;
  skills: Record<string, Record<string, number | null>>;
}

export interface CalibrationCurve {
  model: string;
  label: string;
  overall: (number | null)[];
  by_horizon: Record<string, (number | null)[]>;
}

export interface WeeklyCovSeries {
  model: string;
  label: string;
  all: (number | null)[];
  h8: (number | null)[];
}

export interface PitModelEntry {
  model: string;
  label: string;
  h1?: number[];
  h4?: number[];
  h8?: number[];
  h8_2024?: number[];
  h8_active?: number[];
  [key: string]: string | number[] | undefined;
}

export interface PrModelEntry {
  model: string;
  label: string;
  horizons: Record<string, { pr_auc: number | null; precision: (number | null)[] }>;
}

export interface ErrorRegimeRow {
  regime: string;
  label: string;
  n: number;
  share: number | null;
  wis: number | null;
  wis_log: number | null;
  mae: number | null;
  bias_cases: number | null;
  cov90: number | null;
  under90: number | null;
  over90: number | null;
}

export interface NotableWeek {
  week: number;
  description: string;
  obs_total: number;
  pred_total: number;
  under90_share: number | null;
}

export interface DiagnosticsData {
  reference: string;
  stratified: Record<string, StratifiedRecord[]>;
  uf_skill: {
    models: { model: string; label: string }[];
    ufs: UfSkillItem[];
  };
  calibration: {
    nominals: number[];
    curves: CalibrationCurve[];
    weekly_2024: {
      origins: number[];
      series: WeeklyCovSeries[];
    };
  };
  pit: {
    bins: number;
    models: PitModelEntry[];
  };
  pr_curves: {
    recall: number[];
    prevalence?: Record<string, number | null>;
    models: PrModelEntry[];
  };
  error_analysis: {
    regimes_by_horizon: Record<string, ErrorRegimeRow[]>;
    notable_weeks: NotableWeek[];
  };
}

export type MetricType = "cases" | "q50" | "outbreak" | "residual";
export type FanMode = "series" | "origin";
export type StratumDim = "by_band" | "by_season" | "by_region" | "by_year_turn";
export type CalMode = "overall" | "weekly2024";

export interface LiveHorizon {
  "q0.05": number[];
  "q0.5": number[];
  "q0.95": number[];
  p: number[];
}

export interface LiveData {
  origin: number;
  geocodes: number[];
  targets: Record<string, number>;
  nowcast_weeks: number;
  obs_weeks?: number[];
  obs?: number[][];
  [key: string]: unknown;
}

export interface KpiSnapshot {
  casesLabel: string;
  predLabel: string;
  casesText: string;
  predText: string;
  resText: string;
  resBiasClass: "pos-bias" | "neg-bias" | null;
  outbreakText: string;
}

export const MODEL_COLORS: Record<string, string> = {
  naive: "#898781",
  naive_level: "#52514e",
  lgbm: "#6250d6",
  lgbm_cqr: "#2a78d6",
  lgbm_cqr_level: "#1baf7a",
  lgbm_aci: "#eb6834",
  lgbm_aci_level: "#eda100",
  lgbm_immunity: "#2a78d6",
  gnn: "#e87ba4",
  gnn_nograph: "#008300",
};

export const SHORT_LABELS: Record<string, string> = {
  naive: "Ingênuo",
  naive_level: "Ingênuo por nível",
  lgbm: "LightGBM",
  lgbm_cqr_level: "LightGBM + CQR nível",
  lgbm_aci: "LightGBM + ACI",
  lgbm_immunity: "LightGBM + Imunidade",
  gnn: "ST-GNN",
  gnn_nograph: "Rede sem grafo",
};
