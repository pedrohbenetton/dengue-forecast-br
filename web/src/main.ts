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

type MetricType = "cases" | "q50" | "outbreak";

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

function formatPercent(val: number | null | undefined, digits: number = 1): string {
  if (val == null || isNaN(val)) return "—";
  return (
    (val * 100).toLocaleString("pt-BR", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }) + "%"
  );
}

class App {
  private map: maplibregl.Map | null = null;
  private isMapLoaded: boolean = false;
  private isPlaying: boolean = false;
  private playTimer: number | null = null;

  private activeMetric: MetricType = "cases";
  private activeHorizon: number = 1;
  private currentWeekIndex: number = 0;

  private meta: MapMeta | null = null;
  private municipalities: Municipality[] = [];
  private muniByGeocode: Map<number, Municipality> = new Map();
  private geocodeStrings: string[] = [];
  private geocodeIndexMap: Map<number, number> = new Map();

  private binCache: Map<string, Uint8Array> = new Map();

  private selectedMunicipality: Municipality | null = null;
  private currentFanData: FanData | null = null;
  private live: LiveData | null = null;
  private liveIndex: Map<number, number> = new Map();
  private fanFetchController: AbortController | null = null;

  constructor() {
    this.initTabs();
    this.initControls();
    this.initSearch();
    this.initMap();
    this.renderBenchmarkTable();
    this.loadLive();
  }

  private async loadLive(): Promise<void> {
    try {
      const res = await fetch("./data/live.json");
      if (!res.ok) return; // sem previsão ao vivo publicada ainda
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
    box.innerHTML = `<h4>Previsão atual (origem ${formatEpiweek(this.live.origin)})</h4>
      <table class="live-table"><thead><tr><th>Horizonte</th><th>Mediana</th>
      <th>Intervalo 90%</th><th>P(surto)</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="live-note">As ${this.live.nowcast_weeks} semanas mais recentes usam o nowcast do
      InfoDengue, porque as notificações ainda estão chegando.</p>`;
  }

  private initTabs(): void {
    const tabs = document.querySelectorAll<HTMLButtonElement>(".nav-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");

        const targetId = tab.getAttribute("data-tab");
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
        if (targetId) {
          document.getElementById(targetId)?.classList.add("active");
          if (targetId === "map-view" && this.map) {
            this.map.resize();
          }
        }
      });
    });
  }

  private initControls(): void {
    const casesEl = document.getElementById("drawer-kpi-cases");
    const predEl = document.getElementById("drawer-kpi-pred");
    const outbreakEl = document.getElementById("drawer-kpi-outbreak");
    if (casesEl) casesEl.textContent = "—";
    if (predEl) predEl.textContent = "—";
    if (outbreakEl) outbreakEl.textContent = "—";

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

      const geojson = feature(topoData, topoData.objects.BRMU) as unknown as GeoJSON.FeatureCollection;

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

      this.map.on("mouseenter", "municipios-fill", () => {
        if (this.map) this.map.getCanvas().style.cursor = "pointer";
      });

      this.map.on("mouseleave", "municipios-fill", () => {
        if (this.map) this.map.getCanvas().style.cursor = "";
      });

      const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
      if (this.meta && this.meta.weeks.length > 0) {
        const initIdx = this.meta.weeks.indexOf(this.meta.latest_week);
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
    const isOutbreak = this.activeMetric === "outbreak";
    const incScale = this.meta.inc_scale || 32.0;

    for (let g = 0; g < nGeo; g++) {
      const code = bin[offset + g];
      const val =
        code === 255
          ? null
          : isOutbreak
            ? (code / 254) * 100
            : Math.expm1(code / incScale);

      this.map.setFeatureState(
        { source: "municipios", id: this.geocodeStrings[g] },
        { v: val }
      );
    }
  }

  private updateWeekLabel(): void {
    const label = document.getElementById("current-epiweek");
    if (!label || !this.meta || this.meta.weeks.length === 0) return;
    const week = this.meta.weeks[this.currentWeekIndex];
    label.textContent = formatEpiweek(week);
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
      if (!this.meta) {
        this.stopPlay();
        return;
      }

      if (this.currentWeekIndex >= this.meta.weeks.length - 1) {
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
    }, 400);
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
    const isOutbreak = this.activeMetric === "outbreak";
    const titleText = isOutbreak
      ? "Probabilidade de Surto (>300/100k)"
      : this.activeMetric === "q50"
        ? `Previsão Mediana h=${this.activeHorizon} (por 100 mil)`
        : "Casos Observados (por 100 mil)";

    const titleEl = document.getElementById("legend-title");
    if (titleEl) {
      titleEl.textContent = titleText;
    }

    const scaleContainer = document.querySelector<HTMLElement>(".legend-scale");
    if (scaleContainer) {
      if (isOutbreak) {
        scaleContainer.innerHTML = `
          <div class="legend-bar" style="background: linear-gradient(to right, #f2f5fb, #c6d8f0, #7fa8dc, #3c6fc0, #173a7a);"></div>
          <div class="legend-labels">
            <span>0%</span>
            <span>10%</span>
            <span>30%</span>
            <span>60%</span>
            <span>90%+</span>
          </div>
        `;
      } else {
        scaleContainer.innerHTML = `
          <div class="legend-bar" style="background: linear-gradient(to right, #f7f4e9, #fde3a7, #f9b26b, #ee7a47, #c8402c, #7a1020);"></div>
          <div class="legend-labels">
            <span>0</span>
            <span>10</span>
            <span>50</span>
            <span>100</span>
            <span>300</span>
            <span>1.000+</span>
          </div>
        `;
      }
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
    const casesEl = document.getElementById("drawer-kpi-cases");
    const predEl = document.getElementById("drawer-kpi-pred");
    const outbreakEl = document.getElementById("drawer-kpi-outbreak");

    if (nameEl) nameEl.textContent = `${muni.name} — ${muni.uf}`;
    if (metaEl) {
      metaEl.textContent = `Geocode: ${muni.geocode} | Pop: ${muni.population.toLocaleString("pt-BR")}`;
    }

    if (casesEl) casesEl.textContent = "—";
    if (predEl) predEl.textContent = "—";
    if (outbreakEl) outbreakEl.textContent = "—";

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
      if (casesEl) casesEl.textContent = "—";
      if (predEl) predEl.textContent = "—";
      if (outbreakEl) outbreakEl.textContent = "—";
      this.renderFanChartError("Erro ao carregar dados de previsão do município.");
    }
  }

  private async updateDrawerKpisAndFanChart(): Promise<void> {
    const drawer = document.getElementById("municipality-drawer");
    if (!drawer || !drawer.classList.contains("open") || !this.selectedMunicipality) {
      return;
    }

    const casesEl = document.getElementById("drawer-kpi-cases");
    const predEl = document.getElementById("drawer-kpi-pred");
    const outbreakEl = document.getElementById("drawer-kpi-outbreak");

    if (predEl?.previousElementSibling) {
      predEl.previousElementSibling.textContent =
        `Previsão h=${this.activeHorizon} (mediana)`;
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
        // Ignora falha de rede pontual
      }
    }

    if (this.selectedMunicipality?.geocode !== currentGeocode) return;

    if (this.currentFanData && this.meta) {
      const targetWeek = this.meta.weeks[this.currentWeekIndex];
      const weekIdx = this.currentFanData.weeks.indexOf(targetWeek);
      if (weekIdx >= 0) {
        const obsVal = this.currentFanData.obs[weekIdx];
        if (casesEl) casesEl.textContent = formatNumber(obsVal, 0);

        const hKey = `h${this.activeHorizon}`;
        const fanH = this.currentFanData[hKey] as FanQuantiles | undefined;
        const predVal = fanH?.["q0.5"]?.[weekIdx];
        if (predEl) predEl.textContent = formatNumber(predVal, 0);
      } else {
        if (casesEl) casesEl.textContent = "—";
        if (predEl) predEl.textContent = "—";
      }
    } else {
      if (casesEl) casesEl.textContent = "—";
      if (predEl) predEl.textContent = "—";
    }

    const gIdx = this.geocodeIndexMap.get(this.selectedMunicipality.geocode);
    if (outbreakEl) {
      if (pBin && gIdx !== undefined && this.meta && this.meta.geocodes.length > 0) {
        const offset = this.currentWeekIndex * this.meta.geocodes.length;
        const code = pBin[offset + gIdx];
        if (code === undefined || code === 255) {
          outbreakEl.textContent = "—";
        } else {
          outbreakEl.textContent = formatPercent(code / 254, 0);
        }
      } else {
        outbreakEl.textContent = "—";
      }
    }

    if (this.currentFanData) {
      this.renderFanChart(this.currentFanData);
    }
  }

  private renderFanChart(data: FanData): void {
    const svg = document.getElementById("fanchart-svg");
    if (!svg) return;

    const n = data.weeks.length;
    if (n === 0) {
      svg.innerHTML = "";
      return;
    }

    const width = 500;
    const height = 240;
    const padLeft = 45;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 25;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const getX = (index: number): number => {
      if (n <= 1) return padLeft + plotWidth / 2;
      return padLeft + (index / (n - 1)) * plotWidth;
    };

    const hKey = `h${this.activeHorizon}`;
    const quantiles = data[hKey] as FanQuantiles | undefined;
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

    const getY = (val: number): number => {
      const clamped = Math.max(0, val);
      return padTop + plotHeight - (clamped / yMaxVal) * plotHeight;
    };

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
      if (start !== null) {
        segments.push([start, valid.length - 1]);
      }
      return segments;
    };

    const elements: string[] = [];

    // 3 rótulos no eixo Y e linhas de grade
    const yTicks = [0, Math.round(yMaxVal / 2), yMaxVal];
    yTicks.forEach((v) => {
      const y = getY(v);
      elements.push(
        `<line x1="${padLeft}" y1="${y.toFixed(1)}" ` +
          `x2="${width - padRight}" y2="${y.toFixed(1)}" ` +
          `stroke="#e1e0d9" stroke-width="1" />`
      );
      elements.push(
        `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" ` +
          `text-anchor="end" font-size="10" fill="#898781">${formatNumber(v, 0)}</text>`
      );
    });

    // Eixo X com o ano no início de cada ano
    const axisY = padTop + plotHeight;
    elements.push(
      `<line x1="${padLeft}" y1="${axisY}" ` +
        `x2="${width - padRight}" y2="${axisY}" ` +
        `stroke="#c3c2b7" stroke-width="1" />`
    );

    for (let i = 0; i < n; i++) {
      const yr = Math.floor(data.weeks[i] / 100);
      const prevYr = i > 0 ? Math.floor(data.weeks[i - 1] / 100) : null;
      if ((prevYr !== null && yr !== prevYr) || (i === 0 && (data.weeks[0] % 100) <= 5)) {
        const x = getX(i);
        elements.push(
          `<line x1="${x.toFixed(1)}" y1="${axisY}" ` +
            `x2="${x.toFixed(1)}" y2="${axisY + 4}" ` +
            `stroke="#c3c2b7" stroke-width="1" />`
        );
        elements.push(
          `<text x="${x.toFixed(1)}" y="${axisY + 15}" ` +
            `text-anchor="middle" font-size="10" fill="#898781">${yr}</text>`
        );
      }
    }

    // Faixa 90% (q0.05 - q0.95)
    const valid90 = data.weeks.map(
      (_, i) => q05[i] != null && !isNaN(q05[i]!) && q95[i] != null && !isNaN(q95[i]!)
    );
    getSegments(valid90).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(q95[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) {
          d += ` L ${getX(i).toFixed(1)},${getY(q95[i]!).toFixed(1)}`;
        }
        for (let i = e; i >= s; i--) {
          d += ` L ${getX(i).toFixed(1)},${getY(q05[i]!).toFixed(1)}`;
        }
        d += " Z";
        elements.push(`<path d="${d}" fill="rgba(42, 120, 214, 0.15)" stroke="none" />`);
      }
    });

    // Faixa 50% (q0.25 - q0.75)
    const valid50 = data.weeks.map(
      (_, i) => q25[i] != null && !isNaN(q25[i]!) && q75[i] != null && !isNaN(q75[i]!)
    );
    getSegments(valid50).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(q75[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) {
          d += ` L ${getX(i).toFixed(1)},${getY(q75[i]!).toFixed(1)}`;
        }
        for (let i = e; i >= s; i--) {
          d += ` L ${getX(i).toFixed(1)},${getY(q25[i]!).toFixed(1)}`;
        }
        d += " Z";
        elements.push(`<path d="${d}" fill="rgba(42, 120, 214, 0.30)" stroke="none" />`);
      }
    });

    // Mediana (#2a78d6)
    const validMed = data.weeks.map((_, i) => q50[i] != null && !isNaN(q50[i]!));
    getSegments(validMed).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(q50[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) {
          d += ` L ${getX(i).toFixed(1)},${getY(q50[i]!).toFixed(1)}`;
        }
        elements.push(`<path d="${d}" fill="none" stroke="#2a78d6" stroke-width="1.8" />`);
      } else {
        elements.push(
          `<circle cx="${getX(s).toFixed(1)}" cy="${getY(q50[s]!).toFixed(1)}" ` +
            `r="1.5" fill="#2a78d6" />`
        );
      }
    });

    // Série observada (linha preta fina)
    const validObs = data.weeks.map((_, i) => obs[i] != null && !isNaN(obs[i]!));
    getSegments(validObs).forEach(([s, e]) => {
      if (e > s) {
        let d = `M ${getX(s).toFixed(1)},${getY(obs[s]!).toFixed(1)}`;
        for (let i = s + 1; i <= e; i++) {
          d += ` L ${getX(i).toFixed(1)},${getY(obs[i]!).toFixed(1)}`;
        }
        elements.push(`<path d="${d}" fill="none" stroke="#0b0b0b" stroke-width="1.2" />`);
      } else {
        elements.push(
          `<circle cx="${getX(s).toFixed(1)}" cy="${getY(obs[s]!).toFixed(1)}" ` +
            `r="1.2" fill="#0b0b0b" />`
        );
      }
    });

    // Linha vertical tracejada na semana selecionada
    if (this.meta && this.meta.weeks.length > 0) {
      const selectedWeek = this.meta.weeks[this.currentWeekIndex];
      const selIdx = data.weeks.indexOf(selectedWeek);
      const targetIdx = selIdx >= 0 ? selIdx : this.currentWeekIndex;
      if (targetIdx >= 0 && targetIdx < n) {
        const xSel = getX(targetIdx);
        elements.push(
          `<line x1="${xSel.toFixed(1)}" y1="${padTop}" ` +
            `x2="${xSel.toFixed(1)}" y2="${axisY}" ` +
            `stroke="#eb6834" stroke-width="1.5" stroke-dasharray="4,3" />`
        );
      }
    }

    svg.innerHTML = elements.join("\n");
  }

  private renderFanChartError(message: string): void {
    const svg = document.getElementById("fanchart-svg");
    if (!svg) return;
    svg.innerHTML =
      `<text x="250" y="120" text-anchor="middle" font-size="12" fill="#898781">` +
      `${message}</text>`;
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
          return (
            "<tr>" +
            `<td>${w(row.label)}</td>` +
            `<td>${w(formatNumber(row.wis?.["1"], 2))}</td>` +
            `<td>${w(formatNumber(row.wis?.["2"], 2))}</td>` +
            `<td>${w(formatNumber(row.wis?.["4"], 2))}</td>` +
            `<td>${w(formatNumber(row.wis?.["8"], 2))}</td>` +
            `<td>${w(formatPercent(row.cov90, 1))}</td>` +
            `<td>${w(formatNumber(row.pr_auc_8, 3))}</td>` +
            "</tr>"
          );
        })
        .join("");
    } catch {
      tbody.innerHTML =
        '<tr><td colspan="7">Erro ao carregar dados de benchmark.</td></tr>';
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new App();
});
