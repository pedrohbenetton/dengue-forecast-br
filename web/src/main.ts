import { feature } from "topojson-client";
import maplibregl from "maplibre-gl";

interface Municipality {
  geocode: number;
  name: string;
  uf: string;
  region: string;
  population: number;
  normName?: string;
}

interface MapMeta {
  weeks: number[];
  geocodes: number[];
  horizons: number[];
  inc_scale: number;
  res_scale?: number;
  res_zero?: number;
  latest_week: number;
}

interface FanQuantiles {
  "q0.05": (number | null)[];
  "q0.25": (number | null)[];
  "q0.5": (number | null)[];
  "q0.75": (number | null)[];
  "q0.95": (number | null)[];
}

interface FanData {
  weeks: number[];
  obs: (number | null)[];
  h1?: FanQuantiles;
  h2?: FanQuantiles;
  h4?: FanQuantiles;
  h8?: FanQuantiles;
  [key: string]: FanQuantiles | number[] | (number | null)[] | undefined;
}

interface BenchmarkRow {
  model: string;
  label: string;
  wis: Record<string, number | null>;
  rel_wis_8: number | null;
  cov90: number | null;
  pr_auc_8: number | null;
}

interface BenchmarksData {
  rows: BenchmarkRow[];
  reference: string;
  source: string;
}

interface StratifiedRecord {
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

interface UfSkillItem {
  uf_code: number;
  uf: string;
  region: string;
  skills: Record<string, Record<string, number | null>>;
}

interface CalibrationCurve {
  model: string;
  label: string;
  overall: (number | null)[];
  by_horizon: Record<string, (number | null)[]>;
}

interface WeeklyCovSeries {
  model: string;
  label: string;
  all: (number | null)[];
  h8: (number | null)[];
}

interface PitModelEntry {
  model: string;
  label: string;
  h1?: number[];
  h4?: number[];
  h8?: number[];
  h8_2024?: number[];
  h8_active?: number[];
  [key: string]: string | number[] | undefined;
}

interface PrModelEntry {
  model: string;
  label: string;
  horizons: Record<string, { pr_auc: number | null; precision: (number | null)[] }>;
}

interface ErrorRegimeRow {
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

interface NotableWeek {
  week: number;
  description: string;
  obs_total: number;
  pred_total: number;
  under90_share: number | null;
}

interface DiagnosticsData {
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

type MetricType = "cases" | "q50" | "outbreak" | "residual";
type FanMode = "series" | "origin";
type StratumDim = "by_band" | "by_season" | "by_region" | "by_year_turn";
type CalMode = "overall" | "weekly2024";

interface LiveHorizon {
  "q0.05": number[];
  "q0.5": number[];
  "q0.95": number[];
  p: number[];
}

interface LiveData {
  origin: number;
  geocodes: number[];
  targets: Record<string, number>;
  nowcast_weeks: number;
  [key: string]: unknown;
}

const MODEL_COLORS: Record<string, string> = {
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

const SHORT_LABELS: Record<string, string> = {
  naive: "Ingênuo",
  naive_level: "Ingênuo por nível",
  lgbm: "LightGBM",
  lgbm_cqr_level: "LightGBM + CQR nível",
  lgbm_aci: "LightGBM + ACI",
  lgbm_immunity: "LightGBM + Imunidade",
  gnn: "ST-GNN",
  gnn_nograph: "Rede sem grafo",
};

function normalizeText(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

function formatEpiweek(week: number): string {
  const s = String(week);
  if (s.length >= 6) {
    return `${s.slice(0, 4)}-${s.slice(4)}`;
  }
  return s;
}

function formatNumber(val: number | null | undefined, digits: number): string {
  if (val == null || isNaN(val)) return "—";
  return val.toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatSigned(val: number | null | undefined, digits: number): string {
  if (val == null || isNaN(val)) return "—";
  const sign = val > 0 ? "+" : "";
  return sign + formatNumber(val, digits);
}

function formatPercent(val: number | null | undefined, digits: number = 1): string {
  if (val == null || isNaN(val)) return "—";
  return (
    (val * 100).toLocaleString("pt-BR", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }) + "%"
  );
}

function formatStratumName(dim: StratumDim, raw: string): string {
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

function skillBgColor(relWis: number | null | undefined): string {
  if (relWis == null || isNaN(relWis)) return "transparent";
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

class App {
  private map: maplibregl.Map | null = null;
  private isMapLoaded: boolean = false;
  private isPlaying: boolean = false;
  private playTimer: number | null = null;

  private activeMetric: MetricType = "cases";
  private activeHorizon: number = 1;
  private currentWeekIndex: number = 0;
  private fanMode: FanMode = "series";

  private stratumDim: StratumDim = "by_band";
  private stratumHorizon: number = 8;
  private calMode: CalMode = "overall";
  private pitSlice: string = "h8";
  private pitModel: string = "lgbm_cqr_level";
  private prHorizon: number = 8;
  private ufModel: string = "lgbm_cqr_level";
  private errHorizon: number = 8;

  private meta: MapMeta | null = null;
  private municipalities: Municipality[] = [];
  private muniByGeocode: Map<number, Municipality> = new Map();
  private geocodeStrings: string[] = [];
  private geocodeIndexMap: Map<number, number> = new Map();

  private binCache: Map<string, Uint8Array> = new Map();
  private currentMapValues: (number | null)[] = [];

  private selectedMunicipality: Municipality | null = null;
  private currentFanData: FanData | null = null;
  private live: LiveData | null = null;
  private liveIndex: Map<number, number> = new Map();
  private diagnostics: DiagnosticsData | null = null;
  private fanFetchController: AbortController | null = null;

  constructor() {
    this.initTabs();
    this.initControls();
    this.initDiagnosticControls();
    this.initSearch();
    this.initMap();
    this.renderBenchmarkTable();
    this.loadDiagnostics();
    this.loadLive();
  }

  private switchTab(targetId: string): void {
    document.querySelectorAll<HTMLButtonElement>(".nav-tab").forEach((t) => {
      t.classList.toggle("active", t.getAttribute("data-tab") === targetId);
    });
    document.querySelectorAll(".tab-content").forEach((c) => {
      c.classList.toggle("active", c.id === targetId);
    });
    if (targetId === "map-view" && this.map) {
      this.map.resize();
    }
  }

  private initTabs(): void {
    const tabs = document.querySelectorAll<HTMLButtonElement>(".nav-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetId = tab.getAttribute("data-tab");
        if (targetId) this.switchTab(targetId);
      });
    });
  }

  private async loadLive(): Promise<void> {
    try {
      const res = await fetch("./data/live.json");
      if (!res.ok) return;
      this.live = (await res.json()) as LiveData;
      this.liveIndex = new Map(this.live.geocodes.map((g, i) => [g, i]));
    } catch {
      this.live = null;
    }
  }

  private renderLive(muni: Municipality): void {
    const box = document.getElementById("drawer-live");
    if (!box) return;
    const idx = this.live ? this.liveIndex.get(muni.geocode) : undefined;
    if (!this.live || idx === undefined) {
      box.innerHTML = "";
      return;
    }
    const fmt = (v: number) => Math.round(v).toLocaleString("pt-BR");
    const rows = Object.keys(this.live.targets)
      .map((h) => {
        const block = this.live?.[`h${h}`] as LiveHorizon | undefined;
        if (!block) return "";
        const target = formatEpiweek(this.live?.targets[h] ?? 0);
        return `<tr><td>${h} sem (${target})</td><td>${fmt(block["q0.5"][idx])}</td>
          <td>${fmt(block["q0.05"][idx])}–${fmt(block["q0.95"][idx])}</td>
          <td>${formatPercent(block.p[idx], 0)}</td></tr>`;
      })
      .join("");
    box.innerHTML = `<h4>Previsão ao vivo (origem ${formatEpiweek(this.live.origin)})</h4>
      <table class="live-table"><thead><tr><th>Horizonte</th><th>Mediana</th>
      <th>Intervalo 90%</th><th>P(surto)</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="live-note">As ${this.live.nowcast_weeks} semanas mais recentes usam o nowcast do
      InfoDengue (<code>casos_est</code>) para compensar o atraso de digitação.</p>`;
  }

  private initControls(): void {
    const metricBtns = document.querySelectorAll<HTMLButtonElement>(".btn-toggle");
    metricBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        metricBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const newMetric = (btn.getAttribute("data-metric") as MetricType) || "cases";
        if (this.activeMetric !== newMetric) {
          this.activeMetric = newMetric;
          this.updateLayerStyles();
          this.updateLegend();
          this.loadActiveBinaryAndUpdateMap();
          this.updateDrawerKpisAndFanChart();
        }
      });
    });

    const horizonBtns = document.querySelectorAll<HTMLButtonElement>(".btn-pill");
    horizonBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        horizonBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const newHorizon = parseInt(btn.getAttribute("data-horizon") || "1", 10);
        if (this.activeHorizon !== newHorizon) {
          this.activeHorizon = newHorizon;
          if (this.activeMetric !== "cases") {
            this.updateLegend();
            this.loadActiveBinaryAndUpdateMap();
          }
          this.updateDrawerKpisAndFanChart();
        }
      });
    });

    const fanModeBtns = document.querySelectorAll<HTMLButtonElement>(
      "#fan-mode-selector .mini-pill"
    );
    fanModeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        fanModeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.fanMode = (btn.getAttribute("data-fan-mode") as FanMode) || "series";
        const descEl = document.getElementById("fanchart-desc");
        if (descEl) {
          descEl.textContent =
            this.fanMode === "origin"
              ? "Cone multi-horizonte (h = 1, 2, 4, 8) exatamente como emitido na semana selecionada sobre a trajetória futura real."
              : "Linha preta: casos observados. Faixas azuis: intervalos calibrados de 50% e 90% no horizonte selecionado.";
        }
        if (this.currentFanData) {
          this.renderFanChart(this.currentFanData);
        }
      });
    });

    const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
    if (slider) {
      slider.addEventListener("input", () => {
        this.currentWeekIndex = parseInt(slider.value, 10);
        this.updateWeekLabel();
        this.updateMapState();
        this.updateDrawerKpisAndFanChart();
      });
    }

    const playBtn = document.getElementById("btn-play-timeline");
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        this.togglePlay();
      });
    }

    const closeBtn = document.getElementById("btn-close-drawer");
    const drawer = document.getElementById("municipality-drawer");
    if (closeBtn && drawer) {
      closeBtn.addEventListener("click", () => {
        drawer.classList.remove("open");
        this.selectedMunicipality = null;
        this.currentFanData = null;
      });
    }
  }

  private initDiagnosticControls(): void {
    document
      .querySelectorAll<HTMLButtonElement>("#stratum-dim-selector .mini-pill")
      .forEach((btn) => {
        btn.addEventListener("click", () => {
          document
            .querySelectorAll("#stratum-dim-selector .mini-pill")
            .forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          this.stratumDim = (btn.getAttribute("data-dim") as StratumDim) || "by_band";
          this.renderStratifiedTable();
        });
      });

    document
      .querySelectorAll<HTMLButtonElement>("#stratum-h-selector .mini-pill")
      .forEach((btn) => {
        btn.addEventListener("click", () => {
          document
            .querySelectorAll("#stratum-h-selector .mini-pill")
            .forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          this.stratumHorizon = parseInt(btn.getAttribute("data-sh") || "8", 10);
          this.renderStratifiedTable();
        });
      });

    document.querySelectorAll<HTMLButtonElement>("#cal-mode-selector .mini-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        document
          .querySelectorAll("#cal-mode-selector .mini-pill")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.calMode = (btn.getAttribute("data-cal") as CalMode) || "overall";
        this.renderCalibrationChart();
      });
    });

    const pitSliceEl = document.getElementById("pit-slice-select") as HTMLSelectElement | null;
    if (pitSliceEl) {
      pitSliceEl.addEventListener("change", () => {
        this.pitSlice = pitSliceEl.value;
        this.renderPitChart();
      });
    }

    const pitModelEl = document.getElementById("pit-model-select") as HTMLSelectElement | null;
    if (pitModelEl) {
      pitModelEl.addEventListener("change", () => {
        this.pitModel = pitModelEl.value;
        this.renderPitChart();
      });
    }

    document.querySelectorAll<HTMLButtonElement>("#pr-h-selector .mini-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        document
          .querySelectorAll("#pr-h-selector .mini-pill")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.prHorizon = parseInt(btn.getAttribute("data-prh") || "8", 10);
        this.renderPrChart();
      });
    });

    const ufSelect = document.getElementById("uf-model-select") as HTMLSelectElement | null;
    if (ufSelect) {
      ufSelect.addEventListener("change", () => {
        this.ufModel = ufSelect.value;
        this.renderUfHeatmap();
      });
    }

    document.querySelectorAll<HTMLButtonElement>("#err-h-selector .mini-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        document
          .querySelectorAll("#err-h-selector .mini-pill")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.errHorizon = parseInt(btn.getAttribute("data-errh") || "8", 10);
        this.renderErrorRegimeTable();
      });
    });
  }

  private initMap(): void {
    const container = document.getElementById("map-container");
    if (!container) return;

    this.map = new maplibregl.Map({
      container,
      style: {
        version: 8,
        sources: {},
        layers: [
          {
            id: "background",
            type: "background",
            paint: { "background-color": "#f4f3ee" },
          },
        ],
      },
      center: [-52.0, -14.0],
      zoom: 3.8,
      minZoom: 3.0,
      maxZoom: 12.0,
    });

    this.map.addControl(new maplibregl.NavigationControl(), "top-right");
    new ResizeObserver(() => this.map?.resize()).observe(container);

    this.map.on("load", () => {
      this.isMapLoaded = true;
      this.loadInitialData();
    });
  }

  private async loadInitialData(): Promise<void> {
    try {
      const [topoRes, muniRes, metaRes] = await Promise.all([
        fetch("./data/municipios.topo.json"),
        fetch("./data/municipalities.json"),
        fetch("./data/map/meta.json"),
      ]);

      if (!topoRes.ok || !muniRes.ok || !metaRes.ok) {
        throw new Error("Falha ao carregar metadados ou geometrias.");
      }

      const topoData = await topoRes.json();
      this.municipalities = await muniRes.json();
      this.meta = await metaRes.json();

      this.muniByGeocode.clear();
      this.municipalities.forEach((m) => {
        m.normName = normalizeText(m.name);
        this.muniByGeocode.set(m.geocode, m);
      });

      if (this.meta) {
        this.geocodeStrings = this.meta.geocodes.map((g) => String(g));
        this.geocodeIndexMap = new Map(this.meta.geocodes.map((g, idx) => [g, idx]));
      }

      const geojson = feature(
        topoData,
        topoData.objects.BRMU
      ) as unknown as GeoJSON.FeatureCollection;

      if (!this.map) return;

      this.map.addSource("municipios", {
        type: "geojson",
        data: geojson as any,
        promoteId: "codarea",
      });

      this.map.addLayer({
        id: "municipios-fill",
        type: "fill",
        source: "municipios",
        paint: {
          "fill-color": this.getFillColorExpression() as any,
        },
      });

      this.map.addLayer({
        id: "municipios-line",
        type: "line",
        source: "municipios",
        paint: {
          "line-color": "#ffffff",
          "line-width": 0.2,
          "line-opacity": 0.4,
        },
      });

      this.map.on("click", "municipios-fill", (e) => {
        if (!e.features || e.features.length === 0) return;
        const f = e.features[0];
        const codeVal = f.properties?.codarea || f.id;
        if (!codeVal) return;
        const geocode = parseInt(String(codeVal), 10);
        const muni = this.muniByGeocode.get(geocode);
        if (muni) {
          this.openMunicipalityDrawer(muni);
        }
      });

      const tooltip = document.getElementById("map-hover-tooltip");
      this.map.on("mousemove", "municipios-fill", (e) => {
        if (this.map) this.map.getCanvas().style.cursor = "pointer";
        if (!tooltip || !e.features || e.features.length === 0) return;
        const f = e.features[0];
        const codeVal = f.properties?.codarea || f.id;
        const geocode = codeVal ? parseInt(String(codeVal), 10) : NaN;
        const muni = this.muniByGeocode.get(geocode);
        const gIdx = this.geocodeIndexMap.get(geocode);
        if (!muni || gIdx === undefined) {
          tooltip.classList.add("hidden");
          return;
        }
        const val = this.currentMapValues[gIdx];
        let valText = "sem dado";
        if (val != null && !isNaN(val)) {
          if (this.activeMetric === "outbreak") {
            valText = `P(surto): ${formatNumber(val, 0)}%`;
          } else if (this.activeMetric === "residual") {
            const direction =
              val > 0.08 ? " (subestimou)" : val < -0.08 ? " (superestimou)" : " (alinhado)";
            valText = `Δ log-inc: ${formatSigned(val, 2)}${direction}`;
          } else {
            valText = `${formatNumber(val, 0)} / 100k hab.`;
          }
        }
        tooltip.innerHTML = `<strong>${muni.name} — ${muni.uf}</strong><br/>${valText}`;
        tooltip.style.left = `${e.point.x + 14}px`;
        tooltip.style.top = `${e.point.y + 14}px`;
        tooltip.classList.remove("hidden");
      });

      this.map.on("mouseleave", "municipios-fill", () => {
        if (this.map) this.map.getCanvas().style.cursor = "";
        tooltip?.classList.add("hidden");
      });

      const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
      if (this.meta && this.meta.weeks.length > 0) {
        const defaultPeak = this.meta.weeks.indexOf(202412);
        const initIdx =
          defaultPeak >= 0
            ? defaultPeak
            : Math.max(0, this.meta.weeks.indexOf(this.meta.latest_week));
        this.currentWeekIndex = initIdx >= 0 ? initIdx : this.meta.weeks.length - 1;

        if (slider) {
          slider.min = "0";
          slider.max = String(this.meta.weeks.length - 1);
          slider.value = String(this.currentWeekIndex);
        }
        this.updateWeekLabel();
      }

      this.updateLegend();
      await this.loadActiveBinaryAndUpdateMap();
    } catch (err) {
      console.error("Erro na inicialização dos dados:", err);
    }
  }

  private getFillColorExpression(): any {
    if (this.activeMetric === "outbreak") {
      return [
        "case",
        ["==", ["feature-state", "v"], null],
        "#e9e8e2",
        [
          "interpolate",
          ["linear"],
          ["feature-state", "v"],
          0,
          "#f2f5fb",
          10,
          "#c6d8f0",
          30,
          "#7fa8dc",
          60,
          "#3c6fc0",
          90,
          "#173a7a",
        ],
      ];
    }

    if (this.activeMetric === "residual") {
      return [
        "case",
        ["==", ["feature-state", "v"], null],
        "#e9e8e2",
        [
          "interpolate",
          ["linear"],
          ["feature-state", "v"],
          -2.5,
          "#184a99",
          -1.0,
          "#82b3e8",
          0.0,
          "#f4f3ee",
          1.0,
          "#f59e56",
          2.2,
          "#c8402c",
          3.2,
          "#7a1020",
        ],
      ];
    }

    return [
      "case",
      ["==", ["feature-state", "v"], null],
      "#e9e8e2",
      [
        "interpolate",
        ["linear"],
        ["feature-state", "v"],
        0,
        "#f7f4e9",
        10,
        "#fde3a7",
        50,
        "#f9b26b",
        100,
        "#ee7a47",
        300,
        "#c8402c",
        1000,
        "#7a1020",
      ],
    ];
  }

  private updateLayerStyles(): void {
    if (!this.map || !this.map.getLayer("municipios-fill")) return;
    this.map.setPaintProperty("municipios-fill", "fill-color", this.getFillColorExpression());
  }

  private getActiveBinaryKey(): string {
    if (this.activeMetric === "cases") return "obs";
    if (this.activeMetric === "q50") return `q50_h${this.activeHorizon}`;
    if (this.activeMetric === "residual") return `err_h${this.activeHorizon}`;
    return `p_h${this.activeHorizon}`;
  }

  private async loadActiveBinaryAndUpdateMap(): Promise<void> {
    const key = this.getActiveBinaryKey();
    if (!this.binCache.has(key)) {
      try {
        const res = await fetch(`./data/map/${key}.bin`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();
        this.binCache.set(key, new Uint8Array(buf));
      } catch (err) {
        console.error(`Erro ao carregar mapa binário ${key}.bin:`, err);
        return;
      }
    }
    this.updateMapState();
  }

  private updateMapState(): void {
    if (!this.map || !this.meta || !this.isMapLoaded) return;
    const key = this.getActiveBinaryKey();
    const bin = this.binCache.get(key);
    if (!bin) return;

    const nGeo = this.meta.geocodes.length;
    const offset = this.currentWeekIndex * nGeo;
    const incScale = this.meta.inc_scale || 32.0;
    const resScale = this.meta.res_scale || 32.0;
    const resZero = this.meta.res_zero ?? 127;

    this.currentMapValues = new Array(nGeo);
    for (let g = 0; g < nGeo; g++) {
      const code = bin[offset + g];
      let val: number | null = null;
      if (code !== undefined && code !== 255) {
        if (this.activeMetric === "outbreak") {
          val = (code / 254) * 100;
        } else if (this.activeMetric === "residual") {
          val = (code - resZero) / resScale;
        } else {
          val = Math.expm1(code / incScale);
        }
      }
      this.currentMapValues[g] = val;
      this.map.setFeatureState({ source: "municipios", id: this.geocodeStrings[g] }, { v: val });
    }
  }

  private updateWeekLabel(): void {
    const label = document.getElementById("current-epiweek");
    if (!label || !this.meta || this.meta.weeks.length === 0) return;
    const week = this.meta.weeks[this.currentWeekIndex];
    label.textContent = formatEpiweek(week);
  }

  public jumpToResidualWeek(targetWeek: number, horizon: number = 8): void {
    this.switchTab("map-view");
    this.activeMetric = "residual";
    this.activeHorizon = horizon;

    document.querySelectorAll<HTMLButtonElement>(".btn-toggle").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-metric") === "residual");
    });
    document.querySelectorAll<HTMLButtonElement>(".btn-pill").forEach((b) => {
      b.classList.toggle("active", parseInt(b.getAttribute("data-horizon") || "0", 10) === horizon);
    });

    if (this.meta) {
      const idx = this.meta.weeks.indexOf(targetWeek);
      if (idx >= 0) {
        this.currentWeekIndex = idx;
        const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
        if (slider) slider.value = String(idx);
        this.updateWeekLabel();
      }
    }

    this.updateLayerStyles();
    this.updateLegend();
    this.loadActiveBinaryAndUpdateMap();
    this.updateDrawerKpisAndFanChart();
  }

  private togglePlay(): void {
    const playBtn = document.getElementById("btn-play-timeline");
    const slider = document.getElementById("timeline-range") as HTMLInputElement | null;

    if (this.isPlaying) {
      this.stopPlay();
      return;
    }
    if (!this.meta || this.meta.weeks.length === 0) return;

    if (this.currentWeekIndex >= this.meta.weeks.length - 1) {
      this.currentWeekIndex = 0;
      if (slider) slider.value = "0";
      this.updateWeekLabel();
      this.updateMapState();
      this.updateDrawerKpisAndFanChart();
    }

    this.isPlaying = true;
    if (playBtn) playBtn.textContent = "❚❚";

    this.playTimer = window.setInterval(() => {
      if (!this.meta || this.currentWeekIndex >= this.meta.weeks.length - 1) {
        this.stopPlay();
        return;
      }
      this.currentWeekIndex++;
      if (slider) slider.value = String(this.currentWeekIndex);
      this.updateWeekLabel();
      this.updateMapState();
      this.updateDrawerKpisAndFanChart();
      if (this.currentWeekIndex >= this.meta.weeks.length - 1) {
        this.stopPlay();
      }
    }, 380);
  }

  private stopPlay(): void {
    if (this.playTimer !== null) {
      clearInterval(this.playTimer);
      this.playTimer = null;
    }
    this.isPlaying = false;
    const playBtn = document.getElementById("btn-play-timeline");
    if (playBtn) playBtn.textContent = "▶";
  }

  private updateLegend(): void {
    const titleEl = document.getElementById("legend-title");
    const scaleContainer = document.querySelector<HTMLElement>(".legend-scale");
    if (!titleEl || !scaleContainer) return;

    if (this.activeMetric === "outbreak") {
      titleEl.textContent = `Probabilidade de Surto (>300/100k, h=${this.activeHorizon})`;
      scaleContainer.innerHTML = `
        <div class="legend-bar" style="background: linear-gradient(to right, #f2f5fb, #c6d8f0, #7fa8dc, #3c6fc0, #173a7a);"></div>
        <div class="legend-labels">
          <span>0%</span><span>10%</span><span>30%</span><span>60%</span><span>90%+</span>
        </div>
      `;
    } else if (this.activeMetric === "residual") {
      titleEl.textContent = `Resíduo log1p(inc) em h=${this.activeHorizon} (vermelho = subestimou)`;
      scaleContainer.innerHTML = `
        <div class="legend-bar" style="background: linear-gradient(to right, #184a99, #82b3e8, #f4f3ee, #f59e56, #c8402c, #7a1020);"></div>
        <div class="legend-labels">
          <span>−2,5 (super)</span><span>0</span><span>+1,5</span><span>+3,0+ (sub)</span>
        </div>
      `;
    } else {
      titleEl.textContent =
        this.activeMetric === "q50"
          ? `Previsão Mediana h=${this.activeHorizon} (por 100 mil)`
          : "Casos Observados (por 100 mil)";
      scaleContainer.innerHTML = `
        <div class="legend-bar" style="background: linear-gradient(to right, #f7f4e9, #fde3a7, #f9b26b, #ee7a47, #c8402c, #7a1020);"></div>
        <div class="legend-labels">
          <span>0</span><span>10</span><span>50</span><span>100</span><span>300</span><span>1.000+</span>
        </div>
      `;
    }
  }

  private initSearch(): void {
    const input = document.getElementById("search-input") as HTMLInputElement | null;
    const results = document.getElementById("search-results");
    if (!input || !results) return;

    input.addEventListener("input", () => {
      const q = normalizeText(input.value);
      if (q.length < 2) {
        results.classList.add("hidden");
        results.innerHTML = "";
        return;
      }

      const matches = this.municipalities
        .filter((m) => (m.normName && m.normName.includes(q)) || String(m.geocode).startsWith(q))
        .slice(0, 8);

      if (matches.length === 0) {
        results.classList.add("hidden");
        results.innerHTML = "";
        return;
      }

      results.innerHTML = "";
      matches.forEach((m) => {
        const item = document.createElement("div");
        item.className = "search-item";
        item.textContent = `${m.name} — ${m.uf} (${m.geocode})`;
        item.addEventListener("click", () => {
          this.openMunicipalityDrawer(m);
          results.classList.add("hidden");
          input.value = "";
        });
        results.appendChild(item);
      });
      results.classList.remove("hidden");
    });

    document.addEventListener("click", (e) => {
      if (!input.contains(e.target as Node) && !results.contains(e.target as Node)) {
        results.classList.add("hidden");
      }
    });
  }

  public async openMunicipalityDrawer(muni: Municipality): Promise<void> {
    this.selectedMunicipality = muni;
    const drawer = document.getElementById("municipality-drawer");
    const nameEl = document.getElementById("drawer-muni-name");
    const metaEl = document.getElementById("drawer-muni-meta");

    if (nameEl) nameEl.textContent = `${muni.name} — ${muni.uf}`;
    if (metaEl) {
      metaEl.textContent = `Geocode: ${muni.geocode} | Pop: ${muni.population.toLocaleString("pt-BR")}`;
    }

    drawer?.classList.add("open");
    this.renderLive(muni);

    if (this.fanFetchController) {
      this.fanFetchController.abort();
    }
    this.fanFetchController = new AbortController();

    try {
      const res = await fetch(`./data/fan/${muni.geocode}.json`, {
        signal: this.fanFetchController.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: FanData = await res.json();
      if (this.selectedMunicipality?.geocode !== muni.geocode) return;
      this.currentFanData = data;
      this.updateDrawerKpisAndFanChart();
    } catch (err: any) {
      if (err.name === "AbortError") return;
      if (this.selectedMunicipality?.geocode !== muni.geocode) return;
      this.currentFanData = null;
      this.renderFanChartError("Erro ao carregar dados de previsão do município.");
    }
  }

  private async updateDrawerKpisAndFanChart(): Promise<void> {
    const drawer = document.getElementById("municipality-drawer");
    if (!drawer || !drawer.classList.contains("open") || !this.selectedMunicipality) {
      return;
    }

    const casesEl = document.getElementById("drawer-kpi-cases");
    const predLabelEl = document.getElementById("drawer-kpi-pred-label");
    const predEl = document.getElementById("drawer-kpi-pred");
    const resEl = document.getElementById("drawer-kpi-res");
    const outbreakEl = document.getElementById("drawer-kpi-outbreak");

    if (predLabelEl) {
      predLabelEl.textContent = `Mediana h=${this.activeHorizon}`;
    }

    const currentGeocode = this.selectedMunicipality.geocode;
    const pKey = `p_h${this.activeHorizon}`;
    let pBin = this.binCache.get(pKey);
    if (!pBin) {
      try {
        const res = await fetch(`./data/map/${pKey}.bin`);
        if (res.ok) {
          const buf = await res.arrayBuffer();
          pBin = new Uint8Array(buf);
          this.binCache.set(pKey, pBin);
        }
      } catch {
        // Ignora falha pontual
      }
    }

    if (this.selectedMunicipality?.geocode !== currentGeocode) return;

    if (resEl) {
      resEl.classList.remove("pos-bias", "neg-bias");
    }

    if (this.currentFanData && this.meta) {
      const targetWeek = this.meta.weeks[this.currentWeekIndex];
      const weekIdx = this.currentFanData.weeks.indexOf(targetWeek);
      if (weekIdx >= 0) {
        const obsVal = this.currentFanData.obs[weekIdx];
        if (casesEl) casesEl.textContent = formatNumber(obsVal, 0);

        const fanH = this.currentFanData[`h${this.activeHorizon}`] as FanQuantiles | undefined;
        const predVal = fanH?.["q0.5"]?.[weekIdx];
        if (predEl) predEl.textContent = formatNumber(predVal, 0);

        if (resEl) {
          if (obsVal != null && predVal != null && !isNaN(obsVal) && !isNaN(predVal)) {
            const diff = obsVal - predVal;
            resEl.textContent = formatSigned(diff, 0);
            if (diff > 2) resEl.classList.add("pos-bias");
            else if (diff < -2) resEl.classList.add("neg-bias");
          } else {
            resEl.textContent = "—";
          }
        }
      }
    }

    const gIdx = this.geocodeIndexMap.get(this.selectedMunicipality.geocode);
    if (outbreakEl) {
      if (pBin && gIdx !== undefined && this.meta && this.meta.geocodes.length > 0) {
        const offset = this.currentWeekIndex * this.meta.geocodes.length;
        const code = pBin[offset + gIdx];
        outbreakEl.textContent =
          code === undefined || code === 255 ? "—" : formatPercent(code / 254, 0);
      } else {
        outbreakEl.textContent = "—";
      }
    }

    if (this.currentFanData) {
      this.renderFanChart(this.currentFanData);
    }
  }

  private renderFanChart(data: FanData): void {
    if (this.fanMode === "origin") {
      this.renderOriginConeFanChart(data);
    } else {
      this.renderRollingSeriesFanChart(data);
    }
  }

  private renderRollingSeriesFanChart(data: FanData): void {
    const svg = document.getElementById("fanchart-svg");
    if (!svg) return;
    const n = data.weeks.length;
    if (n === 0) {
      svg.innerHTML = "";
      return;
    }

    const width = 500;
    const height = 245;
    const padLeft = 46;
    const padRight = 14;
    const padTop = 14;
    const padBottom = 26;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const getX = (index: number): number =>
      n <= 1 ? padLeft + plotWidth / 2 : padLeft + (index / (n - 1)) * plotWidth;

    const quantiles = data[`h${this.activeHorizon}`] as FanQuantiles | undefined;
    const q05 = quantiles?.["q0.05"] || [];
    const q25 = quantiles?.["q0.25"] || [];
    const q50 = quantiles?.["q0.5"] || [];
    const q75 = quantiles?.["q0.75"] || [];
    const q95 = quantiles?.["q0.95"] || [];
    const obs = data.obs || [];

    let maxVal = 0;
    for (let i = 0; i < n; i++) {
      const o = obs[i];
      if (o != null && !isNaN(o) && o > maxVal) maxVal = o;
      const q = q95[i];
      if (q != null && !isNaN(q) && q > maxVal) maxVal = q;
    }
    const yMaxVal = maxVal > 0 ? Math.ceil(maxVal * 1.05) : 10;
    const getY = (val: number): number =>
      padTop + plotHeight - (Math.max(0, val) / yMaxVal) * plotHeight;

    const getSegments = (valid: boolean[]): [number, number][] => {
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
    };

    const elements: string[] = [];
    [0, Math.round(yMaxVal / 2), yMaxVal].forEach((v) => {
      const y = getY(v);
      elements.push(
        `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${width - padRight}" y2="${y.toFixed(1)}" stroke="#e1e0d9" stroke-width="1" />`,
        `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#898781">${formatNumber(v, 0)}</text>`
      );
    });

    const axisY = padTop + plotHeight;
    elements.push(
      `<line x1="${padLeft}" y1="${axisY}" x2="${width - padRight}" y2="${axisY}" stroke="#c3c2b7" stroke-width="1" />`
    );

    for (let i = 0; i < n; i++) {
      const yr = Math.floor(data.weeks[i] / 100);
      const prevYr = i > 0 ? Math.floor(data.weeks[i - 1] / 100) : null;
      if ((prevYr !== null && yr !== prevYr) || (i === 0 && data.weeks[0] % 100 <= 5)) {
        const x = getX(i);
        elements.push(
          `<line x1="${x.toFixed(1)}" y1="${axisY}" x2="${x.toFixed(1)}" y2="${axisY + 4}" stroke="#c3c2b7" stroke-width="1" />`,
          `<text x="${x.toFixed(1)}" y="${axisY + 15}" text-anchor="middle" font-size="10" fill="#898781">${yr}</text>`
        );
      }
    }

    const valid90 = data.weeks.map(
      (_, i) => q05[i] != null && !isNaN(q05[i]!) && q95[i] != null && !isNaN(q95[i]!)
    );
    getSegments(valid90).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(q95[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(q95[i]!).toFixed(1)}`;
        for (let i = e; i >= s; i--) d += ` L ${getX(i).toFixed(1)},${getY(q05[i]!).toFixed(1)}`;
        elements.push(`<path d="${d} Z" fill="rgba(42, 120, 214, 0.16)" stroke="none" />`);
      }
    });

    const valid50 = data.weeks.map(
      (_, i) => q25[i] != null && !isNaN(q25[i]!) && q75[i] != null && !isNaN(q75[i]!)
    );
    getSegments(valid50).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(q75[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(q75[i]!).toFixed(1)}`;
        for (let i = e; i >= s; i--) d += ` L ${getX(i).toFixed(1)},${getY(q25[i]!).toFixed(1)}`;
        elements.push(`<path d="${d} Z" fill="rgba(42, 120, 214, 0.32)" stroke="none" />`);
      }
    });

    const validMed = data.weeks.map((_, i) => q50[i] != null && !isNaN(q50[i]!));
    getSegments(validMed).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(q50[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(q50[i]!).toFixed(1)}`;
        elements.push(`<path d="${d}" fill="none" stroke="#2a78d6" stroke-width="1.8" />`);
      }
    });

    const validObs = data.weeks.map((_, i) => obs[i] != null && !isNaN(obs[i]!));
    getSegments(validObs).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(obs[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) d += ` L ${getX(i).toFixed(1)},${getY(obs[i]!).toFixed(1)}`;
        elements.push(`<path d="${d}" fill="none" stroke="#0b0b0b" stroke-width="1.2" />`);
      }
    });

    if (this.meta && this.meta.weeks.length > 0) {
      const selectedWeek = this.meta.weeks[this.currentWeekIndex];
      const selIdx = data.weeks.indexOf(selectedWeek);
      const targetIdx = selIdx >= 0 ? selIdx : this.currentWeekIndex;
      if (targetIdx >= 0 && targetIdx < n) {
        const xSel = getX(targetIdx);
        elements.push(
          `<line x1="${xSel.toFixed(1)}" y1="${padTop}" x2="${xSel.toFixed(1)}" y2="${axisY}" stroke="#eb6834" stroke-width="1.5" stroke-dasharray="4,3" />`
        );
      }
    }

    svg.innerHTML = elements.join("\n");
  }

  private renderOriginConeFanChart(data: FanData): void {
    const svg = document.getElementById("fanchart-svg");
    if (!svg) return;
    const n = data.weeks.length;
    if (n === 0) {
      svg.innerHTML = "";
      return;
    }

    const selectedWeek = this.meta?.weeks[this.currentWeekIndex] ?? data.weeks[0];
    const originIdx = Math.max(0, data.weeks.indexOf(selectedWeek));
    const winStart = Math.max(0, originIdx - 22);
    const winEnd = Math.min(n - 1, originIdx + 18);
    const winLen = Math.max(1, winEnd - winStart);

    const width = 500;
    const height = 245;
    const padLeft = 46;
    const padRight = 26;
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
    if (baseObs != null && !isNaN(baseObs)) {
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
      if (v05 != null && v25 != null && v50 != null && v75 != null && v95 != null) {
        conePts.push({ idx: tIdx, q05: v05, q25: v25, q50: v50, q75: v75, q95: v95 });
      }
    }

    let maxVal = 0;
    for (let i = winStart; i <= winEnd; i++) {
      const o = data.obs[i];
      if (o != null && !isNaN(o) && o > maxVal) maxVal = o;
    }
    conePts.forEach((p) => {
      if (p.q95 > maxVal) maxVal = p.q95;
    });
    const yMaxVal = maxVal > 0 ? Math.ceil(maxVal * 1.08) : 10;
    const getY = (val: number): number =>
      padTop + plotHeight - (Math.max(0, val) / yMaxVal) * plotHeight;

    const elements: string[] = [];
    const axisY = padTop + plotHeight;

    [0, Math.round(yMaxVal / 2), yMaxVal].forEach((v) => {
      const y = getY(v);
      elements.push(
        `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${width - padRight}" y2="${y.toFixed(1)}" stroke="#e1e0d9" stroke-width="1" />`,
        `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#898781">${formatNumber(v, 0)}</text>`
      );
    });
    elements.push(
      `<line x1="${padLeft}" y1="${axisY}" x2="${width - padRight}" y2="${axisY}" stroke="#c3c2b7" stroke-width="1" />`
    );

    const tickStep = Math.max(4, Math.round(winLen / 5));
    for (let i = winStart; i <= winEnd; i += tickStep) {
      const x = getX(i);
      elements.push(
        `<text x="${x.toFixed(1)}" y="${axisY + 15}" text-anchor="middle" font-size="9.5" fill="#898781">${formatEpiweek(data.weeks[i])}</text>`
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
      elements.push(`<path d="${d90} Z" fill="rgba(42, 120, 214, 0.18)" />`);

      let d50 = `M ${getX(conePts[0].idx).toFixed(1)},${getY(conePts[0].q75).toFixed(1)}`;
      for (let k = 1; k < conePts.length; k++) {
        d50 += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q75).toFixed(1)}`;
      }
      for (let k = conePts.length - 1; k >= 0; k--) {
        d50 += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q25).toFixed(1)}`;
      }
      elements.push(`<path d="${d50} Z" fill="rgba(42, 120, 214, 0.36)" />`);

      let dMed = `M ${getX(conePts[0].idx).toFixed(1)},${getY(conePts[0].q50).toFixed(1)}`;
      for (let k = 1; k < conePts.length; k++) {
        dMed += ` L ${getX(conePts[k].idx).toFixed(1)},${getY(conePts[k].q50).toFixed(1)}`;
      }
      elements.push(`<path d="${dMed}" fill="none" stroke="#2a78d6" stroke-width="2.0" />`);
      conePts.slice(1).forEach((p) => {
        elements.push(
          `<circle cx="${getX(p.idx).toFixed(1)}" cy="${getY(p.q50).toFixed(1)}" r="2.8" fill="#2a78d6" />`
        );
      });
    }

    // Série observada até a origem (sólida) e após a origem (tracejada = verdade futura revelada)
    let pastPath = "";
    for (let i = winStart; i <= originIdx; i++) {
      const o = data.obs[i];
      if (o == null || isNaN(o)) continue;
      pastPath += `${pastPath ? " L" : "M"} ${getX(i).toFixed(1)},${getY(o).toFixed(1)}`;
    }
    if (pastPath) {
      elements.push(`<path d="${pastPath}" fill="none" stroke="#0b0b0b" stroke-width="1.6" />`);
    }

    let futPath = "";
    for (let i = originIdx; i <= winEnd; i++) {
      const o = data.obs[i];
      if (o == null || isNaN(o)) continue;
      futPath += `${futPath ? " L" : "M"} ${getX(i).toFixed(1)},${getY(o).toFixed(1)}`;
    }
    if (futPath) {
      elements.push(
        `<path d="${futPath}" fill="none" stroke="#0b0b0b" stroke-width="1.3" stroke-dasharray="3,2.5" />`
      );
    }

    const xOrig = getX(originIdx);
    elements.push(
      `<line x1="${xOrig.toFixed(1)}" y1="${padTop}" x2="${xOrig.toFixed(1)}" y2="${axisY}" stroke="#eb6834" stroke-width="1.4" stroke-dasharray="4,3" />`,
      `<text x="${(xOrig + 4).toFixed(1)}" y="${padTop + 9}" font-size="9.5" font-weight="600" fill="#eb6834">emitido em ${formatEpiweek(data.weeks[originIdx])}</text>`
    );

    svg.innerHTML = elements.join("\n");
  }

  private renderFanChartError(message: string): void {
    const svg = document.getElementById("fanchart-svg");
    if (!svg) return;
    svg.innerHTML = `<text x="250" y="120" text-anchor="middle" font-size="12" fill="#898781">${message}</text>`;
  }

  private async renderBenchmarkTable(): Promise<void> {
    const tbody = document.getElementById("benchmark-table-body");
    if (!tbody) return;

    try {
      const res = await fetch("./data/benchmarks.json");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BenchmarksData = await res.json();

      tbody.innerHTML = data.rows
        .map((row) => {
          const isServed = row.model === "lgbm_cqr_level";
          const w = (val: string) => (isServed ? `<strong>${val}</strong>` : val);
          const relBg = skillBgColor(row.rel_wis_8);
          return (
            "<tr>" +
            `<td>${w(row.label)}</td>` +
            `<td>${w(formatNumber(row.wis?.["1"], 2))}</td>` +
            `<td>${w(formatNumber(row.wis?.["2"], 2))}</td>` +
            `<td>${w(formatNumber(row.wis?.["4"], 2))}</td>` +
            `<td>${w(formatNumber(row.wis?.["8"], 2))}</td>` +
            `<td style="background:${relBg}">${w(row.rel_wis_8 != null ? formatNumber(row.rel_wis_8, 3) + "×" : "—")}</td>` +
            `<td>${w(formatPercent(row.cov90, 1))}</td>` +
            `<td>${w(formatNumber(row.pr_auc_8, 3))}</td>` +
            "</tr>"
          );
        })
        .join("");
    } catch {
      tbody.innerHTML = '<tr><td colspan="8">Erro ao carregar dados de benchmark.</td></tr>';
    }
  }

  private async loadDiagnostics(): Promise<void> {
    try {
      const res = await fetch("./data/diagnostics.json");
      if (!res.ok) return;
      this.diagnostics = (await res.json()) as DiagnosticsData;

      // Popula selects de modelo
      const pitSelect = document.getElementById("pit-model-select") as HTMLSelectElement | null;
      if (pitSelect && this.diagnostics.pit.models.length > 0) {
        pitSelect.innerHTML = this.diagnostics.pit.models
          .map(
            (m) =>
              `<option value="${m.model}" ${m.model === this.pitModel ? "selected" : ""}>${SHORT_LABELS[m.model] || m.label}</option>`
          )
          .join("");
        if (!this.diagnostics.pit.models.some((m) => m.model === this.pitModel)) {
          this.pitModel = this.diagnostics.pit.models[0].model;
        }
      }

      const ufSelect = document.getElementById("uf-model-select") as HTMLSelectElement | null;
      if (ufSelect && this.diagnostics.uf_skill.models.length > 0) {
        ufSelect.innerHTML = this.diagnostics.uf_skill.models
          .map(
            (m) =>
              `<option value="${m.model}" ${m.model === this.ufModel ? "selected" : ""}>${SHORT_LABELS[m.model] || m.label}</option>`
          )
          .join("");
        if (!this.diagnostics.uf_skill.models.some((m) => m.model === this.ufModel)) {
          this.ufModel = this.diagnostics.uf_skill.models[0].model;
        }
      }

      this.renderStratifiedTable();
      this.renderCalibrationChart();
      this.renderPitChart();
      this.renderPrChart();
      this.renderUfHeatmap();
      this.renderErrorRegimeTable();
      this.renderNotableWeeks();
    } catch (err) {
      console.error("Erro ao carregar diagnostics.json:", err);
    }
  }

  private renderStratifiedTable(): void {
    const wrap = document.getElementById("stratified-table-wrap");
    if (!wrap || !this.diagnostics) return;
    const rows = (this.diagnostics.stratified[this.stratumDim] || []).filter(
      (r) => r.horizon === this.stratumHorizon
    );
    if (rows.length === 0) {
      wrap.innerHTML = "<p class='card-sub'>Sem dados estratificados para este recorte.</p>";
      return;
    }

    const strata: string[] = [];
    rows.forEach((r) => {
      if (!strata.includes(r.stratum)) strata.push(r.stratum);
    });

    const featured = [
      "naive",
      "naive_level",
      "lgbm",
      "lgbm_cqr_level",
      "lgbm_immunity",
      "gnn",
      "gnn_nograph",
    ];
    const models = featured.filter((m) => rows.some((r) => r.model === m));

    const headerCells = strata
      .map((s) => `<th>${formatStratumName(this.stratumDim, s)}</th>`)
      .join("");

    const bodyRows = models
      .map((m) => {
        const sample = rows.find((r) => r.model === m);
        const label = sample?.label || m;
        const isServed = m === "lgbm_cqr_level";
        const cells = strata
          .map((s) => {
            const rec = rows.find((r) => r.model === m && r.stratum === s);
            if (!rec || rec.rel_wis == null) return "<td>—</td>";
            const bg = skillBgColor(rec.rel_wis);
            const covTxt = rec.cov90 != null ? ` <small style="color:#52514e">(${formatPercent(rec.cov90, 0)})</small>` : "";
            const mainTxt = `${formatNumber(rec.rel_wis, 2)}×`;
            return `<td style="background:${bg}">${isServed ? `<strong>${mainTxt}</strong>` : mainTxt}${covTxt}</td>`;
          })
          .join("");
        return `<tr><td>${isServed ? `<strong>${label}</strong>` : label}</td>${cells}</tr>`;
      })
      .join("");

    wrap.innerHTML = `
      <table class="benchmark-table">
        <thead><tr><th>Modelo (WIS rel. | Cob. 90%)</th>${headerCells}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>
    `;
  }

  private renderCalibrationChart(): void {
    const svg = document.getElementById("calibration-svg");
    const leg = document.getElementById("calibration-legend");
    if (!svg || !leg || !this.diagnostics) return;

    const width = 460;
    const height = 280;
    const padL = 48;
    const padR = 18;
    const padT = 16;
    const padB = 34;
    const pw = width - padL - padR;
    const ph = height - padT - padB;
    const elements: string[] = [];

    if (this.calMode === "overall") {
      const nominals = this.diagnostics.calibration.nominals;
      const getX = (v: number) => padL + ((v - 0.45) / 0.55) * pw;
      const getY = (v: number) => padT + ph - ((v - 0.45) / 0.55) * ph;

      [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((t) => {
        const y = getY(t);
        elements.push(
          `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="#e1e0d9" />`,
          `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#898781">${Math.round(t * 100)}%</text>`
        );
      });
      nominals.forEach((n) => {
        const x = getX(n);
        elements.push(
          `<text x="${x.toFixed(1)}" y="${padT + ph + 16}" text-anchor="middle" font-size="10" fill="#898781">${Math.round(n * 100)}%</text>`
        );
      });
      elements.push(
        `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="#52514e">Cobertura nominal</text>`,
        `<line x1="${getX(0.45).toFixed(1)}" y1="${getY(0.45).toFixed(1)}" x2="${getX(1.0).toFixed(1)}" y2="${getY(1.0).toFixed(1)}" stroke="#898781" stroke-dasharray="4,3" />`
      );

      const curves = this.diagnostics.calibration.curves;
      curves.forEach((c) => {
        const color = MODEL_COLORS[c.model] || "#0b0b0b";
        let path = "";
        nominals.forEach((nom, i) => {
          const val = c.overall[i];
          if (val == null) return;
          const x = getX(nom);
          const y = getY(val);
          path += `${path ? " L" : "M"} ${x.toFixed(1)},${y.toFixed(1)}`;
          elements.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${color}" />`);
        });
        if (path) {
          elements.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2" />`);
        }
      });

      leg.innerHTML = curves
        .map(
          (c) =>
            `<span class="legend-chip"><span class="legend-swatch" style="background:${MODEL_COLORS[c.model] || "#0b0b0b"}"></span>${SHORT_LABELS[c.model] || c.label}</span>`
        )
        .join("");
    } else {
      const wk = this.diagnostics.calibration.weekly_2024;
      const origins = wk.origins;
      const n = origins.length;
      if (n === 0) {
        svg.innerHTML = `<text x="230" y="140" text-anchor="middle" font-size="11" fill="#898781">Série semanal indisponível.</text>`;
        return;
      }
      const getX = (i: number) => padL + (i / Math.max(1, n - 1)) * pw;
      const getY = (v: number) => padT + ph - ((Math.max(0.5, Math.min(1.0, v)) - 0.5) / 0.5) * ph;

      [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((t) => {
        const y = getY(t);
        const is90 = Math.abs(t - 0.9) < 1e-6;
        elements.push(
          `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="${is90 ? "#898781" : "#e1e0d9"}" ${is90 ? 'stroke-dasharray="4,3"' : ""} />`,
          `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#898781">${Math.round(t * 100)}%</text>`
        );
      });
      for (let i = 0; i < n; i += 10) {
        elements.push(
          `<text x="${getX(i).toFixed(1)}" y="${padT + ph + 16}" text-anchor="middle" font-size="9.5" fill="#898781">${formatEpiweek(origins[i])}</text>`
        );
      }
      elements.push(
        `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="#52514e">Semana de origem na temporada 2023–24 (h=8)</text>`
      );

      wk.series.forEach((s) => {
        const color = MODEL_COLORS[s.model] || "#0b0b0b";
        let path = "";
        s.h8.forEach((v, i) => {
          if (v == null) return;
          path += `${path ? " L" : "M"} ${getX(i).toFixed(1)},${getY(v).toFixed(1)}`;
        });
        if (path) {
          elements.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="1.8" />`);
        }
      });

      leg.innerHTML = wk.series
        .map(
          (s) =>
            `<span class="legend-chip"><span class="legend-swatch" style="background:${MODEL_COLORS[s.model] || "#0b0b0b"}"></span>${SHORT_LABELS[s.model] || s.label}</span>`
        )
        .join("");
    }

    svg.innerHTML = elements.join("\n");
  }

  private renderPitChart(): void {
    const svg = document.getElementById("pit-svg");
    const cap = document.getElementById("pit-caption");
    if (!svg || !this.diagnostics) return;

    const entry = this.diagnostics.pit.models.find((m) => m.model === this.pitModel);
    const values = (entry?.[this.pitSlice] as number[] | undefined) || [];
    if (values.length === 0) {
      svg.innerHTML = `<text x="230" y="140" text-anchor="middle" font-size="11" fill="#898781">Histograma PIT indisponível.</text>`;
      return;
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
    [0, 0.5, 1.0, 1.5, 2.0].filter((t) => t <= yMax + 0.05).forEach((t) => {
      const y = getY(t);
      const isOne = Math.abs(t - 1.0) < 1e-6;
      elements.push(
        `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="${isOne ? "#eb6834" : "#e1e0d9"}" stroke-width="${isOne ? "1.4" : "1"}" ${isOne ? 'stroke-dasharray="4,3"' : ""} />`,
        `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#898781">${formatNumber(t, 1)}</text>`
      );
    });

    const k = values.length;
    const barW = pw / k;
    const color = MODEL_COLORS[this.pitModel] || "#2a78d6";
    values.forEach((v, i) => {
      const x = padL + i * barW + 3;
      const y = getY(v);
      const h = padT + ph - y;
      elements.push(
        `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(barW - 6).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" fill="${color}" fill-opacity="0.78" rx="2" />`
      );
      if (i % 2 === 0 || i === k - 1) {
        const edge = (i + 1) / k;
        elements.push(
          `<text x="${(padL + (i + 1) * barW).toFixed(1)}" y="${padT + ph + 15}" text-anchor="middle" font-size="9.5" fill="#898781">${formatNumber(edge, 1)}</text>`
        );
      }
    });
    elements.push(
      `<text x="${padL}" y="${padT + ph + 15}" text-anchor="middle" font-size="9.5" fill="#898781">0,0</text>`,
      `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="#52514e">Quantil realizado u = F(y) (densidade relativa)</text>`
    );

    svg.innerHTML = elements.join("\n");

    if (cap) {
      const tailRight = values[values.length - 1];
      if (this.pitSlice === "h8_2024") {
        cap.textContent = `No choque de 2023–24, a última faixa (0,9–1,0) atinge densidade ${formatNumber(tailRight, 2)}× a uniforme: assinatura clássica de subestimação na subida epidêmica.`;
      } else if (this.pitSlice === "h8_active") {
        cap.textContent = `Restringindo a semanas ativas (≥10 casos), remove-se o efeito das semanas zeradas; desvios nas bordas indicam subcobertura residual nas caudas.`;
      } else {
        cap.textContent = `O excesso à esquerda (0,0–0,2) reflete semanas com zero casos onde quantis contínuos previstos ficam entre 0 e 1; o centro ligeiramente abaixo de 1,0 indica intervalos conservadores.`;
      }
    }
  }

  private renderPrChart(): void {
    const svg = document.getElementById("pr-svg");
    const leg = document.getElementById("pr-legend");
    if (!svg || !leg || !this.diagnostics) return;

    const recall = this.diagnostics.pr_curves.recall;
    const hStr = String(this.prHorizon);
    const prev = this.diagnostics.pr_curves.prevalence?.[hStr] ?? 0.0326;

    const width = 460;
    const height = 280;
    const padL = 44;
    const padR = 16;
    const padT = 16;
    const padB = 34;
    const pw = width - padL - padR;
    const ph = height - padT - padB;

    const getX = (r: number) => padL + r * pw;
    const getY = (p: number) => padT + ph - p * ph;

    const elements: string[] = [];
    [0, 0.25, 0.5, 0.75, 1.0].forEach((t) => {
      const y = getY(t);
      elements.push(
        `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="#e1e0d9" />`,
        `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#898781">${Math.round(t * 100)}%</text>`
      );
      const x = getX(t);
      elements.push(
        `<text x="${x.toFixed(1)}" y="${padT + ph + 15}" text-anchor="middle" font-size="10" fill="#898781">${Math.round(t * 100)}%</text>`
      );
    });

    const yPrev = getY(prev);
    elements.push(
      `<line x1="${padL}" y1="${yPrev.toFixed(1)}" x2="${width - padR}" y2="${yPrev.toFixed(1)}" stroke="#898781" stroke-dasharray="4,3" />`,
      `<text x="${width - padR - 4}" y="${(yPrev - 4).toFixed(1)}" text-anchor="end" font-size="9" fill="#898781">prevalência (${formatPercent(prev, 1)})</text>`,
      `<text x="${padL + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="#52514e">Recall (sensibilidade aos surtos &gt; 300/100k)</text>`
    );

    const legendItems: string[] = [];
    this.diagnostics.pr_curves.models.forEach((m) => {
      const block = m.horizons[hStr];
      if (!block) return;
      const color = MODEL_COLORS[m.model] || "#0b0b0b";
      let path = "";
      block.precision.forEach((p, idx) => {
        if (p == null || isNaN(p)) return;
        path += `${path ? " L" : "M"} ${getX(recall[idx]).toFixed(1)},${getY(p).toFixed(1)}`;
      });
      if (path) {
        elements.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2" />`);
      }
      legendItems.push(
        `<span class="legend-chip"><span class="legend-swatch" style="background:${color}"></span>${SHORT_LABELS[m.model] || m.label}: <strong>${formatNumber(block.pr_auc, 3)}</strong></span>`
      );
    });

    svg.innerHTML = elements.join("\n");
    leg.innerHTML = legendItems.join("");
  }

  private renderUfHeatmap(): void {
    const wrap = document.getElementById("uf-heatmap-wrap");
    if (!wrap || !this.diagnostics) return;
    const ufs = this.diagnostics.uf_skill.ufs;
    if (ufs.length === 0) {
      wrap.innerHTML = "<p class='card-sub' style='padding:12px'>Skill por UF indisponível.</p>";
      return;
    }

    const rowsHtml = ufs
      .map((u) => {
        const sk = u.skills[this.ufModel] || {};
        const cells = ["1", "2", "4", "8"]
          .map((h) => {
            const v = sk[h];
            const bg = skillBgColor(v);
            return `<td class="skill-cell" style="background:${bg}">${formatNumber(v, 2)}</td>`;
          })
          .join("");
        return `<tr><td><strong>${u.uf}</strong></td><td>${u.region}</td>${cells}</tr>`;
      })
      .join("");

    wrap.innerHTML = `
      <table class="uf-table">
        <thead><tr><th>UF</th><th>Região</th><th>h=1</th><th>h=2</th><th>h=4</th><th>h=8</th></tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    `;
  }

  private renderErrorRegimeTable(): void {
    const wrap = document.getElementById("error-regime-table-wrap");
    if (!wrap || !this.diagnostics) return;
    const rows =
      this.diagnostics.error_analysis.regimes_by_horizon[String(this.errHorizon)] || [];
    if (rows.length === 0) {
      wrap.innerHTML = "<p class='card-sub'>Sem dados de regime para este horizonte.</p>";
      return;
    }

    const body = rows
      .map((r) => {
        const biasCls =
          r.bias_cases != null && r.bias_cases > 2
            ? "color:#d9383a;font-weight:600"
            : r.bias_cases != null && r.bias_cases < -2
              ? "color:#2a78d6;font-weight:600"
              : "";
        return `
          <tr>
            <td><strong>${r.label}</strong></td>
            <td>${formatPercent(r.share, 1)}</td>
            <td>${formatNumber(r.wis, 2)}</td>
            <td>${formatNumber(r.mae, 1)}</td>
            <td style="${biasCls}">${formatSigned(r.bias_cases, 1)}</td>
            <td>${formatPercent(r.cov90, 1)}</td>
            <td>${formatPercent(r.under90, 1)}</td>
            <td>${formatPercent(r.over90, 1)}</td>
          </tr>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <table class="benchmark-table">
        <thead>
          <tr>
            <th>Regime Epidemiológico</th>
            <th>Fração das Semanas</th>
            <th>WIS</th>
            <th>MAE</th>
            <th>Viés Mediano (casos)</th>
            <th>Cob. 90%</th>
            <th>Subcobertura (&gt;q95)</th>
            <th>Supercobertura (&lt;q05)</th>
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    `;
  }

  private renderNotableWeeks(): void {
    const grid = document.getElementById("notable-weeks-grid");
    if (!grid || !this.diagnostics) return;
    const weeks = this.diagnostics.error_analysis.notable_weeks;
    if (weeks.length === 0) {
      grid.innerHTML = "";
      return;
    }

    grid.innerHTML = weeks
      .map(
        (w) => `
        <div class="notable-week-card">
          <div>
            <div class="notable-week-top">
              <span class="notable-week-badge">SE ${formatEpiweek(w.week)}</span>
              <span class="notable-week-stats">Acima de q95: <strong>${formatPercent(w.under90_share, 1)}</strong></span>
            </div>
            <p class="card-sub" style="margin-top:6px">${w.description}</p>
            <p class="notable-week-stats" style="margin-top:6px">
              Total Brasil: <strong>${w.obs_total.toLocaleString("pt-BR")}</strong> obs vs
              <strong>${w.pred_total.toLocaleString("pt-BR")}</strong> soma das medianas (h=8)
            </p>
          </div>
          <button class="btn-jump-map" data-jump-week="${w.week}">Abrir no mapa de resíduos →</button>
        </div>
      `
      )
      .join("");

    grid.querySelectorAll<HTMLButtonElement>(".btn-jump-map").forEach((btn) => {
      btn.addEventListener("click", () => {
        const wk = parseInt(btn.getAttribute("data-jump-week") || "0", 10);
        if (wk > 0) this.jumpToResidualWeek(wk, 8);
      });
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new App();
});
