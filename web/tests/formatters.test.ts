import { describe, expect, it } from "vitest";
import {
  computeDrawerKpis,
  decodeMapByte,
  escapeHtml,
  formatEpiweek,
  formatNumber,
  formatOutbreakProb,
  formatPercent,
  formatSigned,
  formatStratumName,
  normalizeText,
  searchMunicipalities,
  skillBgColor,
} from "../src/lib/formatters";
import type { FanData, MapMeta, Municipality } from "../src/types";

describe("formatters & pure domain helpers", () => {
  it("normalizes accents, case, and punctuation", () => {
    expect(normalizeText("São Paulo — SP")).toBe("sao paulo sp");
    expect(normalizeText("  Águas de Lindóia - SP (3500501) ")).toBe(
      "aguas de lindoia sp 3500501"
    );
  });

  it("escapes HTML special characters safely", () => {
    expect(escapeHtml(`<script>"alert('&')"</script>`)).toBe(
      "&lt;script&gt;&quot;alert(&#39;&amp;&#39;)&quot;&lt;/script&gt;"
    );
  });

  it("formats epiweeks and handles invalid values", () => {
    expect(formatEpiweek(202412)).toBe("2024-12");
    expect(formatEpiweek(202601)).toBe("2026-01");
    expect(formatEpiweek(null)).toBe("—");
    expect(formatEpiweek(NaN)).toBe("—");
    expect(formatEpiweek(0)).toBe("—");
  });

  it("formats numbers and signed residuals without -0 artifacts", () => {
    expect(formatNumber(75293, 0)).toBe("75.293");
    expect(formatNumber(-0.001, 0)).toBe("0");
    expect(formatNumber(NaN, 2)).toBe("—");
    expect(formatNumber(Infinity, 2)).toBe("—");

    expect(formatSigned(14510.9, 0)).toBe("+14.511");
    expect(formatSigned(-250.4, 0)).toBe("-250");
    expect(formatSigned(-0.2, 0)).toBe("0");
    expect(formatSigned(0.1, 0)).toBe("0");
    expect(formatSigned(null, 1)).toBe("—");
  });

  it("formats percentages and quantile-saturated outbreak probabilities consistently", () => {
    expect(formatPercent(0.908, 1)).toBe("90,8%");
    expect(formatPercent(NaN, 1)).toBe("—");

    // 0.025 from live.json AND 6/254 (=0.0236) from p_h*.bin must match identically!
    expect(formatOutbreakProb(0.025)).toBe("≤ 2,5%");
    expect(formatOutbreakProb(6 / 254)).toBe("≤ 2,5%");
    expect(formatOutbreakProb(0.975)).toBe("≥ 97,5%");
    expect(formatOutbreakProb(0.42)).toBe("42%");
    expect(formatOutbreakProb(null)).toBe("—");
  });

  it("formats stratum names and skill background colors", () => {
    expect(formatStratumName("by_season", "2024")).toBe("2023–24 (recorde)");
    expect(formatStratumName("by_season", "2026")).toBe("2025–26");
    expect(formatStratumName("by_year_turn", "true")).toBe("Cruza virada do ano");
    expect(formatStratumName("by_year_turn", "false")).toBe("Mesmo ano civil");

    expect(skillBgColor(0.73)).toContain("rgba(42, 120, 214,");
    expect(skillBgColor(1.22)).toContain("rgba(217, 56, 58,");
    expect(skillBgColor(1.0)).toBe("transparent");
    expect(skillBgColor(null)).toBe("transparent");
  });

  it("decodes binary map bytes across all four layers", () => {
    expect(decodeMapByte(255, "cases")).toBeNull();
    expect(decodeMapByte(undefined, "q50")).toBeNull();
    expect(decodeMapByte(0, "cases", 32)).toBeCloseTo(0, 6);
    expect(decodeMapByte(127, "residual", 32, 32, 127)).toBeCloseTo(0, 6);
    expect(decodeMapByte(159, "residual", 32, 32, 127)).toBeCloseTo(1.0, 6);
    expect(decodeMapByte(95, "residual", 32, 32, 127)).toBeCloseTo(-1.0, 6);
    expect(decodeMapByte(127, "outbreak")).toBeCloseTo(50.0, 4);
  });

  it("computes deterministic KPI snapshots and prevents stale values when fanData is null", () => {
    const meta: MapMeta = {
      weeks: [202411, 202412],
      geocodes: [3550308, 3552205],
      horizons: [1, 2, 4, 8],
      inc_scale: 32,
      res_scale: 32,
      res_zero: 127,
      latest_week: 202412,
    };

    const emptySnap = computeDrawerKpis({
      fanData: null,
      meta,
      weekIndex: 1,
      horizon: 8,
      pCode: undefined,
    });
    expect(emptySnap).toEqual({
      casesLabel: "Observado (2024-12)",
      predLabel: "Mediana h=8 (2024-12)",
      casesText: "—",
      predText: "—",
      resText: "—",
      resBiasClass: null,
      outbreakText: "—",
    });

    const fan: FanData = {
      weeks: [202411, 202412],
      obs: [4100, 4474],
      h8: {
        "q0.05": [1000, 1200],
        "q0.25": [1800, 1900],
        "q0.5": [2300, 2491.7],
        "q0.75": [3000, 3200],
        "q0.95": [4200, 4800],
      },
    };
    const fullSnap = computeDrawerKpis({
      fanData: fan,
      meta,
      weekIndex: 1,
      horizon: 8,
      pCode: 200,
    });
    expect(fullSnap.casesText).toBe("4.474");
    expect(fullSnap.predText).toBe("2.492");
    expect(fullSnap.resText).toBe("+1.982");
    expect(fullSnap.resBiasClass).toBe("pos-bias");
    expect(fullSnap.outbreakText).toBe("79%");
  });

  it("searches municipalities by name, name+UF, and geocode, ranking exact/pop matches first", () => {
    const list: Municipality[] = [
      {
        geocode: 1303908,
        name: "São Paulo de Olivença",
        uf: "AM",
        region: "Norte",
        population: 34000,
      },
      {
        geocode: 3550308,
        name: "São Paulo",
        uf: "SP",
        region: "Sudeste",
        population: 11451999,
      },
      {
        geocode: 3552205,
        name: "Sorocaba",
        uf: "SP",
        region: "Sudeste",
        population: 738128,
      },
    ];

    // Exact name match must rank São Paulo — SP ahead of São Paulo de Olivença — AM!
    const sp = searchMunicipalities(list, "sao paulo");
    expect(sp[0].geocode).toBe(3550308);

    // Name + UF query works cleanly
    const sor = searchMunicipalities(list, "Sorocaba — SP");
    expect(sor).toHaveLength(1);
    expect(sor[0].geocode).toBe(3552205);

    // Geocode prefix works cleanly
    const byCode = searchMunicipalities(list, "35522");
    expect(byCode).toHaveLength(1);
    expect(byCode[0].name).toBe("Sorocaba");

    expect(searchMunicipalities(list, "s")).toEqual([]);
  });
});
