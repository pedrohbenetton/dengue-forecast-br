import { T } from "./theme";
import {
  MODEL_COLORS,
  SHORT_LABELS,
  type CalMode,
  type DiagnosticsData,
  type FanData,
  type FanQuantiles,
} from "../types";
import { escapeHtml, formatEpiweek, formatNumber, formatPercent } from "./formatters";

function getSegments(valid: boolean[]): [number, number][] {
  const segments: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i < valid.length; i++) {
    if (valid[i]) {
      if (start === null) start = i;
    } else if (start !== null) {
      segments.push([start, i - 1]);
      start = null;
    }
  }
  if (start !== null) segments.push([start, valid.length - 1]);
  return segments;
}

export function buildRollingSeriesSvg(
  data: FanData,
  horizon: number,
  selectedWeek: number | undefined
): string {
  const n = data.weeks.length;
  if (n === 0) return "";

  const width = 500;
  const height = 245;
  const padLeft = 48;
  const padRight = 14;
  const padTop = 14;
  const padBottom = 26;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const getX = (index: number): number =>
    n <= 1 ? padLeft + plotWidth / 2 : padLeft + (index / (n - 1)) * plotWidth;

  const quantiles = data[`h${horizon}`] as FanQuantiles | undefined;
  const q05 = quantiles?.["q0.05"] || [];
  const q25 = quantiles?.["q0.25"] || [];
  const q50 = quantiles?.["q0.5"] || [];
  const q75 = quantiles?.["q0.75"] || [];
  const q95 = quantiles?.["q0.95"] || [];
  const obs = data.obs || [];

  let maxVal = 0;
  for (let i = 0; i < n; i++) {
    const o = obs[i];
    if (o != null && Number.isFinite(o) && o > maxVal) maxVal = o;
    const q = q95[i];
    if (q != null && Number.isFinite(q) && q > maxVal) maxVal = q;
  }
  const yMaxVal = maxVal > 0 ? Math.ceil(maxVal * 1.05) : 10;
  const getY = (val: number): number =>
    padTop + plotHeight - (Math.min(yMaxVal, Math.max(0, val)) / yMaxVal) * plotHeight;

  const elements: string[] = [];
  const ticks = Array.from(new Set([0, Math.round(yMaxVal / 2), yMaxVal]));
  ticks.forEach((v) => {
    const y = getY(v);
    elements.push(
      `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${width - padRight}" y2="${y.toFixed(1)}" stroke="${T.grid}" stroke-width="1" />`,
      `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${T.inkMuted}">${formatNumber(v, 0)}</text>`
    );
  });

  const axisY = padTop + plotHeight;
  elements.push(
    `<line x1="${padLeft}" y1="${axisY}" x2="${width - padRight}" y2="${axisY}" stroke="${T.axis}" stroke-width="1" />`
  );

  for (let i = 0; i < n; i++) {
    const yr = Math.floor(data.weeks[i] / 100);
    const prevYr = i > 0 ? Math.floor(data.weeks[i - 1] / 100) : null;
    if ((prevYr !== null && yr !== prevYr) || (i === 0 && data.weeks[0] % 100 <= 5)) {
      const x = getX(i);
      elements.push(
        `<line x1="${x.toFixed(1)}" y1="${axisY}" x2="${x.toFixed(1)}" y2="${axisY + 4}" stroke="${T.axis}" stroke-width="1" />`,
        `<text x="${x.toFixed(1)}" y="${axisY + 15}" text-anchor="middle" font-size="10" fill="${T.inkMuted}">${yr}</text>`
      );
    }
  }

  const valid90 = data.weeks.map(
    (_, i) =>
      q05[i] != null && Number.isFinite(q05[i]!) && q95[i] != null && Number.isFinite(q95[i]!)
  );
  getSegments(valid90).forEach(([s, e]) => {
    if (e > s) {
      let d = `M ${getX(s).toFixed(1)},${getY(q95[s]!).toFixed(1)}`;
      for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(q95[i]!).toFixed(1)}`;
      for (let i = e; i >= s; i--) d += ` L ${getX(i).toFixed(1)},${getY(q05[i]!).toFixed(1)}`;
      elements.push(`<path d="${d} Z" fill="${T.band90}" stroke="none" />`);
    }
  });

  const valid50 = data.weeks.map(
    (_, i) =>
      q25[i] != null && Number.isFinite(q25[i]!) && q75[i] != null && Number.isFinite(q75[i]!)
  );
  getSegments(valid50).forEach(([s, e]) => {
    if (e > s) {
      let d = `M ${getX(s).toFixed(1)},${getY(q75[s]!).toFixed(1)}`;
      for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(q75[i]!).toFixed(1)}`;
      for (let i = e; i >= s; i--) d += ` L ${getX(i).toFixed(1)},${getY(q25[i]!).toFixed(1)}`;
      elements.push(`<path d="${d} Z" fill="${T.band50}" stroke="none" />`);
    }
  });

  const validMed = data.weeks.map((_, i) => q50[i] != null && Number.isFinite(q50[i]!));
  getSegments(validMed).forEach(([s, e]) => {
    if (e > s) {
      let d = `M ${getX(s).toFixed(1)},${getY(q50[s]!).toFixed(1)}`;
      for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(q50[i]!).toFixed(1)}`;
      elements.push(`<path d="${d}" fill="none" stroke="${T.accent}" stroke-width="1.8" />`);
    } else if (e === s) {
      elements.push(
        `<circle cx="${getX(s).toFixed(1)}" cy="${getY(q50[s]!).toFixed(1)}" r="2.2" fill="${T.accent}" />`
      );
    }
  });

  const validObs = data.weeks.map((_, i) => obs[i] != null && Number.isFinite(obs[i]!));
  getSegments(validObs).forEach(([s, e]) => {
    if (e > s) {
      let d = `M ${getX(s).toFixed(1)},${getY(obs[s]!).toFixed(1)}`;
      for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(obs[i]!).toFixed(1)}`;
      elements.push(`<path d="${d}" fill="none" stroke="${T.ink}" stroke-width="1.2" />`);
    } else if (e === s) {
      elements.push(
        `<circle cx="${getX(s).toFixed(1)}" cy="${getY(obs[s]!).toFixed(1)}" r="2.2" fill="${T.ink}" />`
      );
    }
  });

  if (selectedWeek !== undefined) {
    const selIdx = data.weeks.indexOf(selectedWeek);
    if (selIdx >= 0 && selIdx < n) {
      const xSel = getX(selIdx);
      elements.push(
        `<line x1="${xSel.toFixed(1)}" y1="${padTop}" x2="${xSel.toFixed(1)}" y2="${axisY}" stroke="${T.highlight}" stroke-width="1.5" stroke-dasharray="4,3" />`
      );
    }
  }

  return elements.join("\n");
}

export function buildOriginConeSvg(data: FanData, selectedWeek: number | undefined): string {
  const n = data.weeks.length;
  if (n === 0) return "";

  const targetWk = selectedWeek ?? data.weeks[0];
  const foundIdx = data.weeks.indexOf(targetWk);
  const originIdx = foundIdx >= 0 ? foundIdx : 0;
  const winStart = Math.max(0, originIdx - 22);
  const dataEnd = Math.min(n - 1, originIdx + 18);
  // Reserva sempre espaço visual à direita para o cone ou aviso de fim de backtest
  const plotEnd = Math.max(dataEnd, originIdx + 8);
  const winLen = Math.max(1, plotEnd - winStart);

  const width = 500;
  const height = 245;
  const padLeft = 48;
  const padRight = 20;
  const padTop = 18;
  const padBottom = 26;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const getX = (idx: number): number => padLeft + ((idx - winStart) / winLen) * plotWidth;

  const conePts: {
    idx: number;
    q05: number;
    q25: number;
    q50: number;
    q75: number;
    q95: number;
  }[] = [];
  const baseObs = data.obs[originIdx];
  if (baseObs != null && Number.isFinite(baseObs)) {
    conePts.push({
      idx: originIdx,
      q05: baseObs,
      q25: baseObs,
      q50: baseObs,
      q75: baseObs,
      q95: baseObs,
    });
  }
  for (const h of [1, 2, 4, 8]) {
    const tIdx = originIdx + h;
    if (tIdx >= n) continue;
    const fh = data[`h${h}`] as FanQuantiles | undefined;
    const v05 = fh?.["q0.05"]?.[tIdx];
    const v25 = fh?.["q0.25"]?.[tIdx];
    const v50 = fh?.["q0.5"]?.[tIdx];
    const v75 = fh?.["q0.75"]?.[tIdx];
    const v95 = fh?.["q0.95"]?.[tIdx];
    if (
      v05 != null &&
      v25 != null &&
      v50 != null &&
      v75 != null &&
      v95 != null &&
      Number.isFinite(v05) &&
      Number.isFinite(v95)
    ) {
      conePts.push({ idx: tIdx, q05: v05, q25: v25, q50: v50, q75: v75, q95: v95 });
    }
  }

  let maxVal = 0;
  for (let i = winStart; i <= dataEnd; i++) {
    const o = data.obs[i];
    if (o != null && Number.isFinite(o) && o > maxVal) maxVal = o;
  }
  conePts.forEach((p) => {
    if (p.q95 > maxVal) maxVal = p.q95;
  });
  const yMaxVal = maxVal > 0 ? Math.ceil(maxVal * 1.08) : 10;
  const getY = (val: number): number =>
    padTop + plotHeight - (Math.min(yMaxVal, Math.max(0, val)) / yMaxVal) * plotHeight;

  const elements: string[] = [];
  const axisY = padTop + plotHeight;

  const ticks = Array.from(new Set([0, Math.round(yMaxVal / 2), yMaxVal]));
  ticks.forEach((v) => {
    const y = getY(v);
    elements.push(
      `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${width - padRight}" y2="${y.toFixed(1)}" stroke="${T.grid}" stroke-width="1" />`,
      `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${T.inkMuted}">${formatNumber(v, 0)}</text>`
    );
  });
  elements.push(
    `<line x1="${padLeft}" y1="${axisY}" x2="${width - padRight}" y2="${axisY}" stroke="${T.axis}" stroke-width="1" />`
  );

  const tickStep = Math.max(4, Math.round(winLen / 5));
  for (let i = winStart; i <= dataEnd; i += tickStep) {
    const x = getX(i);
    elements.push(
      `<text x="${x.toFixed(1)}" y="${axisY + 15}" text-anchor="middle" font-size="9.5" fill="${T.inkMuted}">${formatEpiweek(data.weeks[i])}</text>`
    );
  }

  if (conePts.length >= 2) {
    let d90 = `M ${getX(conePts[0].idx).toFixed(1)},${getY(conePts[0].q95).toFixed(1)}`;
    for (let k = 1; k < conePts.length; k++) {
      d90 += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q95).toFixed(1)}`;
    }
    for (let k = conePts.length - 1; k >= 0; k--) {
      d90 += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q05).toFixed(1)}`;
    }
    elements.push(`<path d="${d90} Z" fill="${T.band90}" />`);

    let d50 = `M ${getX(conePts[0].idx).toFixed(1)},${getY(conePts[0].q75).toFixed(1)}`;
    for (let k = 1; k < conePts.length; k++) {
      d50 += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q75).toFixed(1)}`;
    }
    for (let k = conePts.length - 1; k >= 0; k--) {
      d50 += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q25).toFixed(1)}`;
    }
    elements.push(`<path d="${d50} Z" fill="${T.band50}" />`);

    let dMed = `M ${getX(conePts[0].idx).toFixed(1)},${getY(conePts[0].q50).toFixed(1)}`;
    for (let k = 1; k < conePts.length; k++) {
      dMed += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q50).toFixed(1)}`;
    }
    elements.push(`<path d="${dMed}" fill="none" stroke="${T.accent}" stroke-width="2.0" />`);
    conePts.slice(1).forEach((p) => {
      elements.push(
        `<circle cx="${getX(p.idx).toFixed(1)}" cy="${getY(p.q50).toFixed(1)}" r="2.8" fill="${T.accent}" />`
      );
    });
  } else {
    const noteX = (getX(originIdx) + (width - padRight)) / 2;
    const noteY = padTop + plotHeight / 2;
    elements.push(
      `<text x="${noteX.toFixed(1)}" y="${(noteY - 6).toFixed(1)}" text-anchor="middle" font-size="9.5" fill="${T.inkSecondary}">Fim do backtest (${formatEpiweek(data.weeks[originIdx])}):</text>`,
      `<text x="${noteX.toFixed(1)}" y="${(noteY + 8).toFixed(1)}" text-anchor="middle" font-size="9.5" fill="${T.inkSecondary}">veja a Previsão ao vivo abaixo</text>`
    );
  }

  let pastPath = "";
  for (let i = winStart; i <= originIdx; i++) {
    const o = data.obs[i];
    if (o == null || !Number.isFinite(o)) continue;
    pastPath += `${pastPath ? " L" : "M"} ${getX(i).toFixed(1)},${getY(o).toFixed(1)}`;
  }
  if (pastPath) {
    elements.push(`<path d="${pastPath}" fill="none" stroke="${T.ink}" stroke-width="1.6" />`);
  } else if (baseObs != null && Number.isFinite(baseObs)) {
    elements.push(
      `<circle cx="${getX(originIdx).toFixed(1)}" cy="${getY(baseObs).toFixed(1)}" r="2.6" fill="${T.ink}" />`
    );
  }

  let futPath = "";
  for (let i = originIdx; i <= dataEnd; i++) {
    const o = data.obs[i];
    if (o == null || !Number.isFinite(o)) continue;
    futPath += `${futPath ? " L" : "M"} ${getX(i).toFixed(1)},${getY(o).toFixed(1)}`;
  }
  if (futPath && dataEnd > originIdx) {
    elements.push(
      `<path d="${futPath}" fill="none" stroke="${T.ink}" stroke-width="1.3" stroke-dasharray="3,2.5" />`
    );
  }

  const xOrig = getX(originIdx);
  const nearRight = xOrig > width - 138;
  const labelX = nearRight ? xOrig - 5 : xOrig + 5;
  const labelAnchor = nearRight ? "end" : "start";
  elements.push(
    `<line x1="${xOrig.toFixed(1)}" y1="${padTop}" x2="${xOrig.toFixed(1)}" y2="${axisY}" stroke="${T.highlight}" stroke-width="1.4" stroke-dasharray="4,3" />`,
    `<text x="${labelX.toFixed(1)}" y="${padTop + 10}" text-anchor="${labelAnchor}" font-size="9.5" font-weight="600" fill="${T.highlight}">emitido em ${formatEpiweek(data.weeks[originIdx])}</text>`
  );

  return elements.join("\n");
}

export function buildCalibrationSvg(
  diag: DiagnosticsData,
  calMode: CalMode
): { svg: string; legendHtml: string } {
  const width = 460;
  const height = 280;
  const padL = 48;
  const padR = 18;
  const padT = 16;
  const padB = 34;
  const pw = width - padL - padR;
  const ph = height - padT - padB;
  const elements: string[] = [];

  if (calMode === "overall") {
    const nominals = diag.calibration.nominals;
    const getX = (v: number) => padL + ((v - 0.45) / 0.55) * pw;
    const getY = (v: number) => padT + ph - ((v - 0.45) / 0.55) * ph;

    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((t) => {
      const y = getY(t);
      elements.push(
        `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="${T.grid}" />`,
        `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${T.inkMuted}">${Math.round(t * 100)}%</text>`
      );
    });
    nominals.forEach((n) => {
      const x = getX(n);
      elements.push(
        `<text x="${x.toFixed(1)}" y="${padT + ph + 16}" text-anchor="middle" font-size="10" fill="${T.inkMuted}">${Math.round(n * 100)}%</text>`
      );
    });
    elements.push(
      `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="${T.inkSecondary}">Cobertura nominal</text>`,
      `<line x1="${getX(0.45).toFixed(1)}" y1="${getY(0.45).toFixed(1)}" x2="${getX(1.0).toFixed(1)}" y2="${getY(1.0).toFixed(1)}" stroke="${T.inkMuted}" stroke-dasharray="4,3" />`
    );

    const curves = diag.calibration.curves;
    curves.forEach((c) => {
      const color = MODEL_COLORS[c.model] || T.ink;
      let path = "";
      nominals.forEach((nom, i) => {
        const val = c.overall[i];
        if (val == null || !Number.isFinite(val)) return;
        const x = getX(nom);
        const y = getY(val);
        path += `${path ? " L" : "M"} ${x.toFixed(1)},${y.toFixed(1)}`;
        elements.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${color}" />`);
      });
      if (path) {
        elements.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2" />`);
      }
    });

    const legendHtml = curves
      .map(
        (c) =>
          `<span class="legend-chip"><span class="legend-swatch" style="background:${MODEL_COLORS[c.model] || T.ink}"></span>${escapeHtml(SHORT_LABELS[c.model] || c.label)}</span>`
      )
      .join("");
    return { svg: elements.join("\n"), legendHtml };
  }

  const wk = diag.calibration.weekly_2024;
  const origins = wk.origins;
  const n = origins.length;
  if (n === 0) {
    return {
      svg: `<text x="230" y="140" text-anchor="middle" font-size="11" fill="${T.inkMuted}">Série semanal indisponível.</text>`,
      legendHtml: "",
    };
  }
  const getX = (i: number) => padL + (i / Math.max(1, n - 1)) * pw;
  const getY = (v: number) => padT + ph - ((Math.max(0.5, Math.min(1.0, v)) - 0.5) / 0.5) * ph;

  [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((t) => {
    const y = getY(t);
    const is90 = Math.abs(t - 0.9) < 1e-6;
    elements.push(
      `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="${is90 ? T.inkMuted : T.grid}" ${is90 ? 'stroke-dasharray="4,3"' : ""} />`,
      `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${T.inkMuted}">${Math.round(t * 100)}%</text>`
    );
  });
  for (let i = 0; i < n; i += 10) {
    elements.push(
      `<text x="${getX(i).toFixed(1)}" y="${padT + ph + 16}" text-anchor="middle" font-size="9.5" fill="${T.inkMuted}">${formatEpiweek(origins[i])}</text>`
    );
  }
  elements.push(
    `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="${T.inkSecondary}">Semana de origem na temporada 2023–24 (h=8)</text>`
  );

  wk.series.forEach((s) => {
    const color = MODEL_COLORS[s.model] || T.ink;
    let path = "";
    s.h8.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) return;
      path += `${path ? " L" : "M"} ${getX(i).toFixed(1)},${getY(v).toFixed(1)}`;
    });
    if (path) {
      elements.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="1.8" />`);
    }
  });

  const legendHtml = wk.series
    .map(
      (s) =>
        `<span class="legend-chip"><span class="legend-swatch" style="background:${MODEL_COLORS[s.model] || T.ink}"></span>${escapeHtml(SHORT_LABELS[s.model] || s.label)}</span>`
    )
    .join("");
  return { svg: elements.join("\n"), legendHtml };
}

export function buildPitSvg(
  diag: DiagnosticsData,
  pitModel: string,
  pitSlice: string
): { svg: string; caption: string } {
  const entry = diag.pit.models.find((m) => m.model === pitModel);
  const values = ((entry?.[pitSlice] as number[] | undefined) || []).filter((v) =>
    Number.isFinite(v)
  );
  if (values.length === 0) {
    return {
      svg: `<text x="230" y="140" text-anchor="middle" font-size="11" fill="${T.inkMuted}">Histograma PIT indisponível.</text>`,
      caption: "",
    };
  }

  const width = 460;
  const height = 280;
  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 34;
  const pw = width - padL - padR;
  const ph = height - padT - padB;
  const yMax = Math.max(1.6, Math.ceil(Math.max(...values) * 1.15 * 10) / 10);
  const getY = (v: number) => padT + ph - (Math.min(yMax, Math.max(0, v)) / yMax) * ph;

  const elements: string[] = [];
  [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    .filter((t) => t <= yMax + 0.05)
    .forEach((t) => {
      const y = getY(t);
      const isOne = Math.abs(t - 1.0) < 1e-6;
      elements.push(
        `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="${isOne ? T.highlight : T.grid}" stroke-width="${isOne ? "1.4" : "1"}" ${isOne ? 'stroke-dasharray="4,3"' : ""} />`,
        `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${T.inkMuted}">${formatNumber(t, 1)}</text>`
      );
    });

  const k = values.length;
  const barW = pw / k;
  const color = MODEL_COLORS[pitModel] || T.accent;
  values.forEach((v, i) => {
    const x = padL + i * barW + 3;
    const y = getY(v);
    const h = padT + ph - y;
    elements.push(
      `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, barW - 6).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" fill="${color}" fill-opacity="0.78" rx="2" />`
    );
    if (i % 2 === 0 || i === k - 1) {
      const edge = (i + 1) / k;
      elements.push(
        `<text x="${(padL + (i + 1) * barW).toFixed(1)}" y="${padT + ph + 15}" text-anchor="middle" font-size="9.5" fill="${T.inkMuted}">${formatNumber(edge, 1)}</text>`
      );
    }
  });
  elements.push(
    `<text x="${padL}" y="${padT + ph + 15}" text-anchor="middle" font-size="9.5" fill="${T.inkMuted}">0,0</text>`,
    `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="${T.inkSecondary}">Quantil realizado u = F(y) (densidade relativa)</text>`
  );

  const tailRight = values[values.length - 1];
  let caption = "";
  if (pitSlice === "h8_2024") {
    caption = `No choque de 2023–24, a última faixa (0,9–1,0) atinge densidade ${formatNumber(tailRight, 2)}× a uniforme: assinatura clássica de subestimação na subida epidêmica.`;
  } else if (pitSlice === "h8_active") {
    caption = `Restringindo a semanas ativas (≥10 casos), remove-se o efeito das semanas zeradas; desvios nas bordas indicam subcobertura residual nas caudas.`;
  } else {
    caption = `O excesso à esquerda (0,0–0,2) reflete semanas com zero casos onde quantis contínuos previstos ficam entre 0 e 1; o centro ligeiramente abaixo de 1,0 indica intervalos conservadores.`;
  }

  return { svg: elements.join("\n"), caption };
}

export function buildPrSvg(
  diag: DiagnosticsData,
  prHorizon: number
): { svg: string; legendHtml: string } {
  const recall = diag.pr_curves.recall;
  const hStr = String(prHorizon);
  const prev = diag.pr_curves.prevalence?.[hStr] ?? 0.0326;

  const width = 460;
  const height = 280;
  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 34;
  const pw = width - padL - padR;
  const ph = height - padT - padB;

  const getX = (r: number) => padL + Math.min(1, Math.max(0, r)) * pw;
  const getY = (p: number) => padT + ph - Math.min(1, Math.max(0, p)) * ph;

  const elements: string[] = [];
  [0, 0.25, 0.5, 0.75, 1.0].forEach((t) => {
    const y = getY(t);
    elements.push(
      `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="${T.grid}" />`,
      `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${T.inkMuted}">${Math.round(t * 100)}%</text>`
    );
    const x = getX(t);
    elements.push(
      `<text x="${x.toFixed(1)}" y="${padT + ph + 15}" text-anchor="middle" font-size="10" fill="${T.inkMuted}">${Math.round(t * 100)}%</text>`
    );
  });

  const yPrev = getY(prev);
  elements.push(
    `<line x1="${padL}" y1="${yPrev.toFixed(1)}" x2="${width - padR}" y2="${yPrev.toFixed(1)}" stroke="${T.inkMuted}" stroke-dasharray="4,3" />`,
    `<text x="${width - padR - 4}" y="${(yPrev - 4).toFixed(1)}" text-anchor="end" font-size="9" fill="${T.inkMuted}">prevalência (${formatPercent(prev, 1)})</text>`,
    `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="${T.inkSecondary}">Recall (sensibilidade aos surtos &gt; 300/100k)</text>`
  );

  const legendItems: string[] = [];
  diag.pr_curves.models.forEach((m) => {
    const block = m.horizons[hStr];
    if (!block) return;
    const color = MODEL_COLORS[m.model] || T.ink;
    let path = "";
    block.precision.forEach((p, idx) => {
      const rVal = recall[idx];
      if (p == null || !Number.isFinite(p) || rVal == null || !Number.isFinite(rVal)) return;
      path += `${path ? " L" : "M"} ${getX(rVal).toFixed(1)},${getY(p).toFixed(1)}`;
    });
    if (path) {
      elements.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2" />`);
    }
    legendItems.push(
      `<span class="legend-chip"><span class="legend-swatch" style="background:${color}"></span>${escapeHtml(SHORT_LABELS[m.model] || m.label)}: <strong>${formatNumber(block.pr_auc, 3)}</strong></span>`
    );
  });

  return { svg: elements.join("\n"), legendHtml: legendItems.join("") };
}
