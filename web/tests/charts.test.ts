import { T } from "../src/lib/theme";
import { describe, expect, it } from "vitest";
import {
  buildCalibrationSvg,
  buildOriginConeSvg,
  buildPitSvg,
  buildPrSvg,
  buildRollingSeriesSvg,
} from "../src/lib/charts";
import type { DiagnosticsData, FanData } from "../src/types";

function makeSyntheticFan(n: number = 30, zeroAll: boolean = false): FanData {
  const weeks = Array.from({ length: n }, (_, i) => 202401 + i);
  const obs = weeks.map((_, i) => (zeroAll ? 0 : i === 5 ? null : i * 10));
  const mkH = (mult: number) => ({
    "q0.05": weeks.map((_, i) => (zeroAll ? 0 : Math.max(0, i * 6 * mult))),
    "q0.25": weeks.map((_, i) => (zeroAll ? 0 : i * 8 * mult)),
    "q0.5": weeks.map((_, i) => (zeroAll ? 0 : i * 10 * mult)),
    "q0.75": weeks.map((_, i) => (zeroAll ? 0 : i * 12 * mult)),
    "q0.95": weeks.map((_, i) => (zeroAll ? 0 : i * 16 * mult)),
  });
  return {
    weeks,
    obs,
    h1: mkH(1.0),
    h2: mkH(1.05),
    h4: mkH(1.1),
    h8: mkH(1.2),
  };
}

describe("SVG chart builders", () => {
  it("renders rolling series SVG cleanly for normal, zero-inflated, 1-point, and empty series", () => {
    expect(buildRollingSeriesSvg({ weeks: [], obs: [] }, 1, undefined)).toBe("");

    const onePt = makeSyntheticFan(1);
    const svg1 = buildRollingSeriesSvg(onePt, 1, 202401);
    expect(svg1).not.toContain("NaN");
    expect(svg1).not.toContain("undefined");
    expect(svg1).toContain("<circle");

    const zeroFan = makeSyntheticFan(25, true);
    const svgZero = buildRollingSeriesSvg(zeroFan, 8, 202410);
    expect(svgZero).not.toContain("NaN");
    expect(svgZero).toContain("stroke-dasharray");

    const normalFan = makeSyntheticFan(30, false);
    const svgNorm = buildRollingSeriesSvg(normalFan, 4, 202412);
    expect(svgNorm).not.toContain("NaN");
    expect(svgNorm).toContain("<path");
  });

  it("renders origin cone SVG at start, middle, and final week without right-edge clipping", () => {
    const fan = makeSyntheticFan(30, false);

    const startSvg = buildOriginConeSvg(fan, fan.weeks[0]);
    expect(startSvg).not.toContain("NaN");
    expect(startSvg).toContain("emitido em 2024-01");
    expect(startSvg).toContain(T.band90);

    const midSvg = buildOriginConeSvg(fan, fan.weeks[12]);
    expect(midSvg).not.toContain("NaN");
    expect(midSvg).toContain("emitido em 2024-13");

    // Last week of backtest: future h=1..8 exceed n, so end-of-backtest notice must render & anchor end!
    const endSvg = buildOriginConeSvg(fan, fan.weeks[fan.weeks.length - 1]);
    expect(endSvg).not.toContain("NaN");
    expect(endSvg).toContain('text-anchor="end"');
    expect(endSvg).toContain("Previsão ao vivo abaixo");
  });

  it("renders calibration, PIT, and PR diagnostic SVGs across modes and empty fallbacks", () => {
    const diag: DiagnosticsData = {
      reference: "naive_level",
      stratified: {},
      uf_skill: { models: [], ufs: [] },
      calibration: {
        nominals: [0.5, 0.8, 0.9, 0.95],
        curves: [
          {
            model: "lgbm_cqr_level",
            label: "LightGBM + CQR por nível",
            overall: [0.67, 0.86, 0.935, 0.96],
            by_horizon: { "8": [0.64, 0.85, 0.92, 0.95] },
          },
        ],
        weekly_2024: {
          origins: [202341, 202401, 202412],
          series: [
            {
              model: "lgbm_cqr_level",
              label: "LightGBM + CQR por nível",
              all: [0.94, 0.88, 0.82],
              h8: [0.92, 0.8, 0.75],
            },
          ],
        },
      },
      pit: {
        bins: 10,
        models: [
          {
            model: "lgbm_cqr_level",
            label: "LightGBM + CQR por nível",
            h8: [1.4, 1.1, 0.9, 0.85, 0.85, 0.85, 0.9, 0.95, 1.05, 1.15],
            h8_2024: [0.9, 0.8, 0.8, 0.8, 0.8, 0.85, 0.9, 1.0, 1.25, 1.9],
          },
        ],
      },
      pr_curves: {
        recall: [0.0, 0.5, 1.0],
        prevalence: { "8": 0.0326 },
        models: [
          {
            model: "lgbm_cqr_level",
            label: "LightGBM + CQR por nível",
            horizons: {
              "8": { pr_auc: 0.44, precision: [0.92, 0.48, 0.033] },
            },
          },
        ],
      },
      error_analysis: { regimes_by_horizon: {}, notable_weeks: [] },
    };

    const calOverall = buildCalibrationSvg(diag, "overall");
    expect(calOverall.svg).not.toContain("NaN");
    expect(calOverall.legendHtml).toContain("LightGBM + CQR nível");

    const calWk = buildCalibrationSvg(diag, "weekly2024");
    expect(calWk.svg).not.toContain("NaN");
    expect(calWk.svg).toContain("2023-41");

    const pit8 = buildPitSvg(diag, "lgbm_cqr_level", "h8_2024");
    expect(pit8.svg).not.toContain("NaN");
    expect(pit8.caption).toContain("1,90×");

    const pitEmpty = buildPitSvg(diag, "missing_model", "h8");
    expect(pitEmpty.svg).toContain("indisponível");

    const pr8 = buildPrSvg(diag, 8);
    expect(pr8.svg).not.toContain("NaN");
    expect(pr8.legendHtml).toContain("0,440");
  });
});
