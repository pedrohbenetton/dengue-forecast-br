import type {
  FanData,
  FanQuantiles,
  KpiSnapshot,
  MapMeta,
  MetricType,
  Municipality,
  StratumDim,
} from "../types";

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function normalizeText(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[—–\-/,()]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function formatEpiweek(week: number | null | undefined): string {
  if (week == null || !Number.isFinite(week) || week <= 0) return "—";
  const s = String(Math.trunc(week));
  if (s.length >= 6) {
    return `${s.slice(0, 4)}-${s.slice(4)}`;
  }
  return s;
}

export function formatNumber(val: number | null | undefined, digits: number): string {
  if (val == null || !Number.isFinite(val)) return "—";
  const eps = 0.5 * Math.pow(10, -digits);
  const clean = Math.abs(val) < eps ? 0 : val;
  return clean.toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatSigned(val: number | null | undefined, digits: number): string {
  if (val == null || !Number.isFinite(val)) return "—";
  const eps = 0.5 * Math.pow(10, -digits);
  if (Math.abs(val) < eps) {
    return formatNumber(0, digits);
  }
  const sign = val > 0 ? "+" : "";
  return sign + formatNumber(val, digits);
}

export function formatPercent(val: number | null | undefined, digits: number = 1): string {
  if (val == null || !Number.isFinite(val)) return "—";
  return (
    formatNumber(val * 100, digits) + "%"
  );
}

/**
 * Formata a probabilidade de surto (> 300/100k) respeitando a saturação estrutural
 * da grade de quantis [0.025 .. 0.975]: quando q0.975 <= limiar, exceedance_probability
 * satura em 0.025 (código uint8 = 6 -> 2,36%), significando P(surto) <= 2,5%.
 */
export function formatOutbreakProb(prob01: number | null | undefined): string {
  if (prob01 == null || !Number.isFinite(prob01)) return "—";
  if (prob01 <= 0.026) return "≤ 2,5%";
  if (prob01 >= 0.974) return "≥ 97,5%";
  return formatPercent(prob01, 0);
}

export function formatStratumName(dim: StratumDim, raw: string): string {
  if (dim === "by_season") {
    const yr = parseInt(raw, 10);
    if (!isNaN(yr)) {
      const lbl = `${yr - 1}–${String(yr % 100).padStart(2, "0")}`;
      return yr === 2024 ? `${lbl} (recorde)` : lbl;
    }
  }
  if (dim === "by_year_turn") {
    return raw.toLowerCase() === "true" ? "Cruza virada do ano" : "Mesmo ano civil";
  }
  return raw;
}

export function skillBgColor(relWis: number | null | undefined): string {
  if (relWis == null || !Number.isFinite(relWis)) return "transparent";
  if (relWis < 1.0) {
    const intensity = Math.min(1, (1.0 - relWis) / 0.35);
    return `rgba(42, 120, 214, ${(0.06 + 0.32 * intensity).toFixed(2)})`;
  }
  if (relWis > 1.0) {
    const intensity = Math.min(1, (relWis - 1.0) / 0.35);
    return `rgba(217, 56, 58, ${(0.06 + 0.28 * intensity).toFixed(2)})`;
  }
  return "transparent";
}

export function decodeMapByte(
  code: number | undefined,
  metric: MetricType,
  incScale: number = 32.0,
  resScale: number = 32.0,
  resZero: number = 127
): number | null {
  if (code === undefined || code === 255) return null;
  if (metric === "outbreak") {
    return (code / 254) * 100;
  }
  if (metric === "residual") {
    return (code - resZero) / resScale;
  }
  return Math.expm1(code / incScale);
}

export function computeDrawerKpis(params: {
  fanData: FanData | null;
  meta: MapMeta | null;
  weekIndex: number;
  horizon: number;
  pCode: number | undefined;
}): KpiSnapshot {
  const { fanData, meta, weekIndex, horizon, pCode } = params;
  const targetWeek =
    meta && weekIndex >= 0 && weekIndex < meta.weeks.length ? meta.weeks[weekIndex] : undefined;
  const wkSuffix = targetWeek ? ` (${formatEpiweek(targetWeek)})` : "";

  const snapshot: KpiSnapshot = {
    casesLabel: targetWeek ? `Observado${wkSuffix}` : "Observado na SE",
    predLabel: `Mediana h=${horizon}${wkSuffix}`,
    casesText: "—",
    predText: "—",
    resText: "—",
    resBiasClass: null,
    outbreakText: pCode === undefined || pCode === 255 ? "—" : formatOutbreakProb(pCode / 254),
  };

  if (!fanData || targetWeek === undefined) {
    return snapshot;
  }

  const weekIdx = fanData.weeks.indexOf(targetWeek);
  if (weekIdx < 0) {
    return snapshot;
  }

  const obsVal = fanData.obs[weekIdx];
  snapshot.casesText = formatNumber(obsVal, 0);

  const fanH = fanData[`h${horizon}`] as FanQuantiles | undefined;
  const predVal = fanH?.["q0.5"]?.[weekIdx];
  snapshot.predText = formatNumber(predVal, 0);

  if (obsVal != null && predVal != null && Number.isFinite(obsVal) && Number.isFinite(predVal)) {
    const diff = obsVal - predVal;
    snapshot.resText = formatSigned(diff, 0);
    if (diff > 2) snapshot.resBiasClass = "pos-bias";
    else if (diff < -2) snapshot.resBiasClass = "neg-bias";
  }

  return snapshot;
}

export function searchMunicipalities(
  municipalities: Municipality[],
  rawQuery: string,
  limit: number = 8
): Municipality[] {
  const q = normalizeText(rawQuery);
  if (q.length < 2) return [];

  const scored: { muni: Municipality; rank: number }[] = [];
  for (const m of municipalities) {
    const normName = m.normName ?? normalizeText(m.name);
    const normUf = m.uf.toLowerCase();
    const normSearch = m.normSearch ?? `${normName} ${normUf}`;
    const codeStr = String(m.geocode);

    let rank = -1;
    if (codeStr === q || normName === q || normSearch === q) {
      rank = 0;
    } else if (codeStr.startsWith(q)) {
      rank = 1;
    } else if (normName.startsWith(q)) {
      rank = 2;
    } else if (normSearch.startsWith(q)) {
      rank = 3;
    } else if (normSearch.includes(q)) {
      rank = 4;
    } else {
      // Todos os tokens da consulta presentes em nome + UF (ex.: "paulo sp")
      const tokens = q.split(" ");
      if (tokens.length > 1 && tokens.every((tk) => normSearch.includes(tk))) {
        rank = 5;
      }
    }

    if (rank >= 0) {
      scored.push({ muni: m, rank });
    }
  }

  scored.sort((a, b) => {
    if (a.rank !== b.rank) return a.rank - b.rank;
    return b.muni.population - a.muni.population;
  });

  return scored.slice(0, limit).map((s) => s.muni);
}
