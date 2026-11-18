import { feature } from "topojson-client";
import maplibregl from "maplibre-gl";
import {
  SHORT_LABELS,
  type BenchmarksData,
  type CalMode,
  type DiagnosticsData,
  type FanData,
  type FanMode,
  type KpiSnapshot,
  type LiveData,
  type LiveHorizon,
  type MapMeta,
  type MetricType,
  type Municipality,
  type StratumDim,
} from "./types";
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
} from "./lib/formatters";
import {
  buildCalibrationSvg,
  buildOriginConeSvg,
  buildPitSvg,
  buildPrSvg,
  buildRollingSeriesSvg,
} from "./lib/charts";

export interface MapAdapter {
  on(type: string, listener: (ev?: any) => void): void;
  on(type: string, layer: string, listener: (ev: any) => void): void;
  addSource(id: string, source: any): void;
  addLayer(layer: any): void;
  getLayer(id: string): any;
  setPaintProperty(layer: string, name: string, value: any): void;
  setFilter(layer: string, filter: any): void;
  setFeatureState(target: { source: string; id: string }, state: Record<string, any>): void;
  fitBounds(bounds: [number, number, number, number], options?: Record<string, any>): void;
  getCanvas(): { style: { cursor: string } };
  resize(): void;
}

export interface AppOptions {
  mapFactory?: (container: HTMLElement) => MapAdapter;
}

function computeFeatureBounds(geom: GeoJSON.Geometry | null | undefined): [number, number, number, number] | null {
  if (!geom) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const visit = (coords: any): void => {
    if (!Array.isArray(coords) || coords.length === 0) return;
    if (typeof coords[0] === "number" && typeof coords[1] === "number") {
      const x = coords[0];
      const y = coords[1];
      if (Number.isFinite(x) && Number.isFinite(y)) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
      return;
    }
    for (const item of coords) visit(item);
  };

  if ("coordinates" in geom) {
    visit(geom.coordinates);
  } else if (geom.type === "GeometryCollection") {
    geom.geometries.forEach((g) => {
      const sub = computeFeatureBounds(g);
      if (sub) {
        if (sub[0] < minX) minX = sub[0];
        if (sub[1] < minY) minY = sub[1];
        if (sub[2] > maxX) maxX = sub[2];
        if (sub[3] > maxY) maxY = sub[3];
      }
    });
  }

  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null;
}

export class App {
  private map: MapAdapter | null = null;
  private mapFactory?: (container: HTMLElement) => MapAdapter;
  private isMapLoaded: boolean = false;
  private isPlaying: boolean = false;
  private playTimer: number | null = null;

  private activeMetric: MetricType = "cases";
  private activeHorizon: number = 1;
  private currentWeekIndex: number = 0;
  private pendingJumpWeek: number | null = null;
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
  private boundsByGeocode: Map<number, [number, number, number, number]> = new Map();
  private geocodeStrings: string[] = [];
  private geocodeIndexMap: Map<number, number> = new Map();

  private binCache: Map<string, Uint8Array> = new Map();
  private currentMapValues: (number | null)[] = [];
  private mapLoadSeq: number = 0;
  private kpiUpdateSeq: number = 0;

  private selectedMunicipality: Municipality | null = null;
  private currentFanData: FanData | null = null;
  private isFanLoading: boolean = false;
  private live: LiveData | null = null;
  private liveLoaded: boolean = false;
  private liveIndex: Map<number, number> = new Map();
  private diagnostics: DiagnosticsData | null = null;
  private fanFetchController: AbortController | null = null;

  constructor(options?: AppOptions) {
    this.mapFactory = options?.mapFactory;
    this.initTabs();
    this.initControls();
    this.initDiagnosticControls();
    this.initSearch();
    this.initGlobalKeys();
    this.initMap();
    void this.renderBenchmarkTable();
    void this.loadDiagnostics();
    void this.loadLive();
  }

  public destroy(): void {
    this.stopPlay();
    this.fanFetchController?.abort();
  }

  private switchTab(targetId: string): void {
    if (targetId !== "map-view") {
      this.stopPlay();
    }
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
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.live = (await res.json()) as LiveData;
      this.liveIndex = new Map(this.live.geocodes.map((g, i) => [g, i]));
    } catch {
      this.live = null;
      this.liveIndex.clear();
    } finally {
      this.liveLoaded = true;
      if (this.selectedMunicipality) {
        this.renderLive(this.selectedMunicipality);
      }
    }
  }

  private renderLive(muni: Municipality): void {
    const box = document.getElementById("drawer-live");
    if (!box) return;
    if (!this.liveLoaded) {
      box.innerHTML = `<p class="live-note">Carregando previsão ao vivo...</p>`;
      return;
    }
    const idx = this.live ? this.liveIndex.get(muni.geocode) : undefined;
    if (!this.live || idx === undefined) {
      box.innerHTML = "";
      return;
    }
    const fmt = (v: number) => Math.round(v).toLocaleString("pt-BR");
    const rows = Object.keys(this.live.targets)
      .sort((a, b) => Number(a) - Number(b))
      .map((h) => {
        const block = this.live?.[`h${h}`] as LiveHorizon | undefined;
        if (!block) return "";
        const target = formatEpiweek(this.live?.targets[h] ?? 0);
        return `<tr><td>${escapeHtml(h)} sem (${escapeHtml(target)})</td><td>${fmt(block["q0.5"][idx])}</td>
          <td>${fmt(block["q0.05"][idx])}–${fmt(block["q0.95"][idx])}</td>
          <td>${formatOutbreakProb(block.p[idx])}</td></tr>`;
      })
      .join("");
    box.innerHTML = `<h4>Previsão ao vivo (origem ${escapeHtml(formatEpiweek(this.live.origin))})</h4>
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
          this.updateLegend();
          void this.loadActiveBinaryAndUpdateMap();
          void this.updateDrawerKpisAndFanChart();
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
            void this.loadActiveBinaryAndUpdateMap();
          }
          void this.updateDrawerKpisAndFanChart();
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
        void this.updateDrawerKpisAndFanChart();
      });
    }

    const playBtn = document.getElementById("btn-play-timeline");
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        this.togglePlay();
      });
    }

    const closeBtn = document.getElementById("btn-close-drawer");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        this.closeMunicipalityDrawer();
      });
    }
  }

  public closeMunicipalityDrawer(): void {
    const drawer = document.getElementById("municipality-drawer");
    drawer?.classList.remove("open");
    drawer?.setAttribute("aria-hidden", "true");
    this.fanFetchController?.abort();
    this.fanFetchController = null;
    this.isFanLoading = false;
    this.selectedMunicipality = null;
    this.currentFanData = null;
    this.updateSelectedPolygonHighlight(null);
    this.map?.resize();
  }

  private initGlobalKeys(): void {
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      const results = document.getElementById("search-results");
      if (results && !results.classList.contains("hidden")) {
        results.classList.add("hidden");
        return;
      }
      const drawer = document.getElementById("municipality-drawer");
      if (drawer && drawer.classList.contains("open")) {
        this.closeMunicipalityDrawer();
      }
    });
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

    if (this.mapFactory) {
      this.map = this.mapFactory(container);
    } else {
      const realMap = new maplibregl.Map({
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
        attributionControl: false,
      });
      realMap.addControl(new maplibregl.NavigationControl(), "top-right");
      this.map = realMap as unknown as MapAdapter;
    }

    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => this.map?.resize()).observe(container);
    }

    this.map.on("load", () => {
      this.isMapLoaded = true;
      void this.loadInitialData();
    });
  }

  private setMapStatusBanner(
    message: string | null,
    isError: boolean = false,
    onRetry?: () => void
  ): void {
    const banner = document.getElementById("map-status-banner");
    if (!banner) return;
    if (!message) {
      banner.classList.add("hidden");
      banner.classList.remove("error");
      banner.innerHTML = "";
      return;
    }
    banner.classList.remove("hidden");
    banner.classList.toggle("error", isError);
    banner.innerHTML = `<span>${escapeHtml(message)}</span>`;
    if (onRetry) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-retry";
      btn.textContent = "Tentar novamente";
      btn.addEventListener("click", onRetry);
      banner.appendChild(btn);
    }
  }

  private async loadInitialData(): Promise<void> {
    this.setMapStatusBanner("Carregando malha municipal...", false);
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
        m.normSearch = `${m.normName} ${m.uf.toLowerCase()}`;
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

      this.boundsByGeocode.clear();
      for (const feat of geojson.features) {
        const rawCode = feat.properties?.codarea ?? feat.id;
        if (!rawCode) continue;
        const code = parseInt(String(rawCode), 10);
        const b = computeFeatureBounds(feat.geometry);
        if (!isNaN(code) && b) {
          this.boundsByGeocode.set(code, b);
        }
      }

      if (!this.map) return;

      if (!this.map.getLayer("municipios-fill")) {
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
            "fill-color": this.getFillColorExpression(),
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

        this.map.addLayer({
          id: "municipios-selected",
          type: "line",
          source: "municipios",
          filter: ["==", ["get", "codarea"], ""],
          paint: {
            "line-color": "#0b0b0b",
            "line-width": 2.0,
            "line-opacity": 0.95,
          },
        });

        this.map.on("click", "municipios-fill", (e: any) => {
          if (!e.features || e.features.length === 0) return;
          const f = e.features[0];
          const codeVal = f.properties?.codarea || f.id;
          if (!codeVal) return;
          const geocode = parseInt(String(codeVal), 10);
          const muni = this.muniByGeocode.get(geocode);
          if (muni) {
            void this.openMunicipalityDrawer(muni);
          } else if (!isNaN(geocode)) {
            this.openUnindexedPolygonDrawer(geocode, f.properties?.nome || f.properties?.name);
          }
        });

        const tooltip = document.getElementById("map-hover-tooltip");
        const stage = document.getElementById("map-stage");
        this.map.on("mousemove", "municipios-fill", (e: any) => {
          if (this.map) this.map.getCanvas().style.cursor = "pointer";
          if (!tooltip || !e.features || e.features.length === 0) return;
          const f = e.features[0];
          const codeVal = f.properties?.codarea || f.id;
          const geocode = codeVal ? parseInt(String(codeVal), 10) : NaN;
          const muni = this.muniByGeocode.get(geocode);
          const gIdx = this.geocodeIndexMap.get(geocode);
          if (!muni || gIdx === undefined) {
            if (!isNaN(geocode)) {
              tooltip.innerHTML = `<strong>Código IBGE ${geocode}</strong><br/>Sem série no InfoDengue`;
            } else {
              tooltip.classList.add("hidden");
              return;
            }
          } else {
            const val = this.currentMapValues[gIdx];
            let valText = "sem dado";
            if (val != null && Number.isFinite(val)) {
              if (this.activeMetric === "outbreak") {
                valText = `P(surto): ${formatOutbreakProb(val / 100)}`;
              } else if (this.activeMetric === "residual") {
                const direction =
                  val > 0.08 ? " (subestimou)" : val < -0.08 ? " (superestimou)" : " (alinhado)";
                valText = `Δ log-inc: ${formatSigned(val, 2)}${direction}`;
              } else {
                valText = `${formatNumber(val, 0)} / 100k hab.`;
              }
            }
            tooltip.innerHTML = `<strong>${escapeHtml(muni.name)} — ${escapeHtml(muni.uf)}</strong><br/>${escapeHtml(valText)}`;
          }

          tooltip.classList.remove("hidden");
          const stageW = stage?.clientWidth || 800;
          const stageH = stage?.clientHeight || 600;
          const tipW = tooltip.offsetWidth || 180;
          const tipH = tooltip.offsetHeight || 44;
          const px = e.point?.x ?? 0;
          const py = e.point?.y ?? 0;

          let left = px + 14;
          let top = py + 14;
          if (left + tipW + 10 > stageW) {
            left = Math.max(8, px - tipW - 14);
          }
          if (top + tipH + 10 > stageH) {
            top = Math.max(8, py - tipH - 14);
          }
          tooltip.style.left = `${left}px`;
          tooltip.style.top = `${top}px`;
        });

        this.map.on("mouseleave", "municipios-fill", () => {
          if (this.map) this.map.getCanvas().style.cursor = "";
          tooltip?.classList.add("hidden");
        });
      }

      if (this.selectedMunicipality) {
        this.updateSelectedPolygonHighlight(this.selectedMunicipality.geocode);
      }

      const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
      if (this.meta && this.meta.weeks.length > 0) {
        let initIdx = -1;
        if (this.pendingJumpWeek !== null) {
          initIdx = this.meta.weeks.indexOf(this.pendingJumpWeek);
          this.pendingJumpWeek = null;
        }
        if (initIdx < 0) {
          const defaultPeak = this.meta.weeks.indexOf(202412);
          initIdx =
            defaultPeak >= 0
              ? defaultPeak
              : Math.max(0, this.meta.weeks.indexOf(this.meta.latest_week));
        }
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
      if (this.selectedMunicipality) {
        void this.updateDrawerKpisAndFanChart();
      }
    } catch (err) {
      console.error("Erro na inicialização dos dados:", err);
      this.setMapStatusBanner("Falha ao carregar os dados iniciais do mapa.", true, () => {
        void this.loadInitialData();
      });
    }
  }

  private updateSelectedPolygonHighlight(geocode: number | null): void {
    if (!this.map || !this.map.getLayer("municipios-selected")) return;
    this.map.setFilter("municipios-selected", [
      "==",
      ["get", "codarea"],
      geocode !== null ? String(geocode) : "",
    ]);
  }

  private getFillColorExpression(metric: MetricType = this.activeMetric): any {
    if (metric === "outbreak") {
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

    if (metric === "residual") {
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

  private clearMapFeatureValues(): void {
    if (!this.map || !this.meta || !this.isMapLoaded) return;
    const nGeo = this.meta.geocodes.length;
    this.currentMapValues = new Array(nGeo).fill(null);
    for (let g = 0; g < nGeo; g++) {
      this.map.setFeatureState({ source: "municipios", id: this.geocodeStrings[g] }, { v: null });
    }
  }

  private async loadActiveBinaryAndUpdateMap(): Promise<void> {
    const key = this.getActiveBinaryKey();
    const seq = ++this.mapLoadSeq;

    if (!this.binCache.has(key)) {
      this.setMapStatusBanner("Carregando camada espacial...", false);
      try {
        const res = await fetch(`./data/map/${key}.bin`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();
        this.binCache.set(key, new Uint8Array(buf));
      } catch (err) {
        if (seq !== this.mapLoadSeq) return;
        console.error(`Erro ao carregar mapa binário ${key}.bin:`, err);
        this.clearMapFeatureValues();
        this.updateLayerStyles();
        this.setMapStatusBanner(`Erro ao carregar camada (${key}.bin).`, true, () => {
          void this.loadActiveBinaryAndUpdateMap();
        });
        return;
      }
    }

    if (seq !== this.mapLoadSeq) return;
    this.setMapStatusBanner(null);
    // Aplica rampa de cores e estados na mesma batida síncrona (sem flash de cores trocadas)
    this.updateLayerStyles();
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
      const val = decodeMapByte(bin[offset + g], this.activeMetric, incScale, resScale, resZero);
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
    this.stopPlay();
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
    } else {
      this.pendingJumpWeek = targetWeek;
    }

    this.updateLegend();
    if (this.isMapLoaded) {
      void this.loadActiveBinaryAndUpdateMap();
    }
    void this.updateDrawerKpisAndFanChart();
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
      void this.updateDrawerKpisAndFanChart();
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
      void this.updateDrawerKpisAndFanChart();
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

    let currentMatches: Municipality[] = [];
    let activeIdx = -1;

    const selectMatch = (m: Municipality): void => {
      void this.openMunicipalityDrawer(m, { flyTo: true });
      results.classList.add("hidden");
      results.innerHTML = "";
      currentMatches = [];
      activeIdx = -1;
      input.value = "";
    };

    const updateHighlight = (): void => {
      results.querySelectorAll<HTMLElement>(".search-item[data-idx]").forEach((el, idx) => {
        el.classList.toggle("active", idx === activeIdx);
        if (idx === activeIdx) {
          el.scrollIntoView({ block: "nearest" });
        }
      });
    };

    input.addEventListener("input", () => {
      const q = normalizeText(input.value);
      activeIdx = -1;
      if (q.length < 2) {
        currentMatches = [];
        results.classList.add("hidden");
        results.innerHTML = "";
        return;
      }

      currentMatches = searchMunicipalities(this.municipalities, input.value, 8);
      results.innerHTML = "";

      if (currentMatches.length === 0) {
        const empty = document.createElement("div");
        empty.className = "search-empty";
        empty.textContent = "Nenhum município encontrado.";
        results.appendChild(empty);
        results.classList.remove("hidden");
        return;
      }

      currentMatches.forEach((m, idx) => {
        const item = document.createElement("div");
        item.className = "search-item";
        item.setAttribute("data-idx", String(idx));
        item.textContent = `${m.name} — ${m.uf} (${m.geocode})`;
        item.addEventListener("click", () => selectMatch(m));
        results.appendChild(item);
      });
      results.classList.remove("hidden");
    });

    input.addEventListener("keydown", (e) => {
      if (results.classList.contains("hidden") || currentMatches.length === 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        activeIdx = (activeIdx + 1) % currentMatches.length;
        updateHighlight();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        activeIdx = activeIdx <= 0 ? currentMatches.length - 1 : activeIdx - 1;
        updateHighlight();
      } else if (e.key === "Enter") {
        e.preventDefault();
        const chosen = currentMatches[activeIdx >= 0 ? activeIdx : 0];
        if (chosen) selectMatch(chosen);
      }
    });

    document.addEventListener("click", (e) => {
      if (!input.contains(e.target as Node) && !results.contains(e.target as Node)) {
        results.classList.add("hidden");
      }
    });
  }

  private applyKpiSnapshot(snap: KpiSnapshot): void {
    const casesLabelEl = document.getElementById("drawer-kpi-cases-label");
    const casesEl = document.getElementById("drawer-kpi-cases");
    const predLabelEl = document.getElementById("drawer-kpi-pred-label");
    const predEl = document.getElementById("drawer-kpi-pred");
    const resEl = document.getElementById("drawer-kpi-res");
    const outbreakEl = document.getElementById("drawer-kpi-outbreak");

    if (casesLabelEl) casesLabelEl.textContent = snap.casesLabel;
    if (casesEl) casesEl.textContent = snap.casesText;
    if (predLabelEl) predLabelEl.textContent = snap.predLabel;
    if (predEl) predEl.textContent = snap.predText;
    if (resEl) {
      resEl.textContent = snap.resText;
      resEl.classList.remove("pos-bias", "neg-bias");
      if (snap.resBiasClass) {
        resEl.classList.add(snap.resBiasClass);
      }
    }
    if (outbreakEl) outbreakEl.textContent = snap.outbreakText;
  }

  private setFanChartState(
    message: string | null,
    isError: boolean = false,
    onRetry?: () => void
  ): void {
    const stateEl = document.getElementById("fanchart-state");
    const svg = document.getElementById("fanchart-svg");
    if (!stateEl) {
      if (svg && message) {
        svg.innerHTML = `<text x="250" y="120" text-anchor="middle" font-size="12" fill="#898781">${escapeHtml(message)}</text>`;
      }
      return;
    }

    if (!message) {
      stateEl.classList.add("hidden");
      stateEl.classList.remove("error");
      stateEl.innerHTML = "";
      return;
    }

    if (svg) svg.innerHTML = "";
    stateEl.classList.remove("hidden");
    stateEl.classList.toggle("error", isError);
    stateEl.innerHTML = "";

    const msgP = document.createElement("p");
    msgP.textContent = message;
    stateEl.appendChild(msgP);

    if (onRetry) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-retry";
      btn.textContent = "Tentar novamente";
      btn.addEventListener("click", onRetry);
      stateEl.appendChild(btn);
    }
  }

  private openUnindexedPolygonDrawer(geocode: number, rawName?: string): void {
    this.fanFetchController?.abort();
    this.fanFetchController = null;
    this.selectedMunicipality = null;
    this.currentFanData = null;
    this.isFanLoading = false;

    const drawer = document.getElementById("municipality-drawer");
    const drawerBody = drawer?.querySelector<HTMLElement>(".drawer-body");
    const nameEl = document.getElementById("drawer-muni-name");
    const metaEl = document.getElementById("drawer-muni-meta");
    const liveEl = document.getElementById("drawer-live");

    if (nameEl) nameEl.textContent = rawName ? String(rawName) : `Área IBGE ${geocode}`;
    if (metaEl) metaEl.textContent = `Geocode: ${geocode} | Sem série no InfoDengue`;
    if (liveEl) liveEl.innerHTML = "";
    if (drawerBody) drawerBody.scrollTop = 0;

    this.applyKpiSnapshot(
      computeDrawerKpis({
        fanData: null,
        meta: this.meta,
        weekIndex: this.currentWeekIndex,
        horizon: this.activeHorizon,
        pCode: undefined,
      })
    );
    this.setFanChartState("Esta área da malha IBGE não possui série epidemiológica no InfoDengue.");
    drawer?.classList.add("open");
    drawer?.setAttribute("aria-hidden", "false");
    this.updateSelectedPolygonHighlight(geocode);
    this.map?.resize();
  }

  public async openMunicipalityDrawer(
    muni: Municipality,
    options?: { flyTo?: boolean }
  ): Promise<void> {
    this.selectedMunicipality = muni;
    this.currentFanData = null;
    this.isFanLoading = true;

    const drawer = document.getElementById("municipality-drawer");
    const drawerBody = drawer?.querySelector<HTMLElement>(".drawer-body");
    const nameEl = document.getElementById("drawer-muni-name");
    const metaEl = document.getElementById("drawer-muni-meta");

    if (nameEl) nameEl.textContent = `${muni.name} — ${muni.uf}`;
    if (metaEl) {
      metaEl.textContent = `Geocode: ${muni.geocode} | Pop: ${muni.population.toLocaleString("pt-BR")}`;
    }

    // Sempre reinicia o scroll e zera KPIs imediatamente para nunca vazar números do município anterior
    if (drawerBody) {
      drawerBody.scrollTop = 0;
    }
    this.applyKpiSnapshot(
      computeDrawerKpis({
        fanData: null,
        meta: this.meta,
        weekIndex: this.currentWeekIndex,
        horizon: this.activeHorizon,
        pCode: this.readCachedOutbreakCode(muni.geocode, this.activeHorizon, this.currentWeekIndex),
      })
    );
    this.setFanChartState("Carregando série histórica do município...", false);

    const wasOpen = drawer?.classList.contains("open");
    drawer?.classList.add("open");
    drawer?.setAttribute("aria-hidden", "false");
    if (!wasOpen) {
      this.map?.resize();
    }

    this.updateSelectedPolygonHighlight(muni.geocode);
    if (options?.flyTo && this.map) {
      const bounds = this.boundsByGeocode.get(muni.geocode);
      if (bounds) {
        this.map.fitBounds(bounds, { padding: 64, maxZoom: 7.5, duration: 550 });
      }
    }

    this.renderLive(muni);

    // Garante também o carregamento assíncrono de P(surto) para o KPI mesmo antes da série terminar
    void this.ensureOutbreakKpi(muni.geocode, this.activeHorizon, this.currentWeekIndex);

    if (this.fanFetchController) {
      this.fanFetchController.abort();
    }
    const controller = new AbortController();
    this.fanFetchController = controller;

    try {
      const res = await fetch(`./data/fan/${muni.geocode}.json`, {
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: FanData = await res.json();
      if (this.selectedMunicipality?.geocode !== muni.geocode) return;
      this.isFanLoading = false;
      this.currentFanData = data;
      this.setFanChartState(null);
      await this.updateDrawerKpisAndFanChart();
    } catch (err: any) {
      if (err?.name === "AbortError") return;
      if (this.selectedMunicipality?.geocode !== muni.geocode) return;
      this.isFanLoading = false;
      this.currentFanData = null;
      this.applyKpiSnapshot(
        computeDrawerKpis({
          fanData: null,
          meta: this.meta,
          weekIndex: this.currentWeekIndex,
          horizon: this.activeHorizon,
          pCode: this.readCachedOutbreakCode(muni.geocode, this.activeHorizon, this.currentWeekIndex),
        })
      );
      this.setFanChartState(
        "Erro ao carregar dados históricos do município.",
        true,
        () => {
          void this.openMunicipalityDrawer(muni);
        }
      );
    }
  }

  private readCachedOutbreakCode(
    geocode: number,
    horizon: number,
    weekIndex: number
  ): number | undefined {
    const pBin = this.binCache.get(`p_h${horizon}`);
    const gIdx = this.geocodeIndexMap.get(geocode);
    if (!pBin || gIdx === undefined || !this.meta || this.meta.geocodes.length === 0) {
      return undefined;
    }
    const offset = weekIndex * this.meta.geocodes.length;
    return pBin[offset + gIdx];
  }

  private async ensureOutbreakKpi(
    geocode: number,
    horizon: number,
    weekIndex: number
  ): Promise<void> {
    const seq = ++this.kpiUpdateSeq;
    const pKey = `p_h${horizon}`;
    if (!this.binCache.has(pKey)) {
      try {
        const res = await fetch(`./data/map/${pKey}.bin`);
        if (res.ok) {
          const buf = await res.arrayBuffer();
          this.binCache.set(pKey, new Uint8Array(buf));
        }
      } catch {
        // Mantém "—" quando binário indisponível
      }
    }
    if (
      seq !== this.kpiUpdateSeq ||
      this.selectedMunicipality?.geocode !== geocode ||
      this.activeHorizon !== horizon ||
      this.currentWeekIndex !== weekIndex
    ) {
      return;
    }
    const code = this.readCachedOutbreakCode(geocode, horizon, weekIndex);
    const outbreakEl = document.getElementById("drawer-kpi-outbreak");
    if (outbreakEl) {
      outbreakEl.textContent =
        code === undefined || code === 255 ? "—" : formatOutbreakProb(code / 254);
    }
  }

  private async updateDrawerKpisAndFanChart(): Promise<void> {
    const drawer = document.getElementById("municipality-drawer");
    if (!drawer || !drawer.classList.contains("open") || !this.selectedMunicipality) {
      return;
    }

    const geocode = this.selectedMunicipality.geocode;
    const horizon = this.activeHorizon;
    const weekIndex = this.currentWeekIndex;
    const cachedPCode = this.readCachedOutbreakCode(geocode, horizon, weekIndex);

    // Atualização 100% síncrona e determinística dos KPIs e do gráfico
    this.applyKpiSnapshot(
      computeDrawerKpis({
        fanData: this.currentFanData,
        meta: this.meta,
        weekIndex,
        horizon,
        pCode: cachedPCode,
      })
    );

    if (this.currentFanData) {
      this.setFanChartState(null);
      this.renderFanChart(this.currentFanData);
    }

    if (cachedPCode === undefined) {
      await this.ensureOutbreakKpi(geocode, horizon, weekIndex);
    }
  }

  private renderFanChart(data: FanData): void {
    const svg = document.getElementById("fanchart-svg");
    if (!svg) return;
    const selectedWeek = this.meta?.weeks[this.currentWeekIndex];
    svg.innerHTML =
      this.fanMode === "origin"
        ? buildOriginConeSvg(data, selectedWeek)
        : buildRollingSeriesSvg(data, this.activeHorizon, selectedWeek);
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
            `<td>${w(escapeHtml(row.label))}</td>` +
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
      tbody.innerHTML =
        '<tr><td colspan="8">Erro ao carregar dados de benchmark. <button type="button" class="btn-retry" id="btn-retry-bench">Tentar novamente</button></td></tr>';
      document.getElementById("btn-retry-bench")?.addEventListener("click", () => {
        void this.renderBenchmarkTable();
      });
    }
  }

  private renderDiagnosticsError(): void {
    const msgHtml = `<div class="diag-error-box">Erro ao carregar diagnósticos detalhados. <button type="button" class="btn-retry js-retry-diag">Tentar novamente</button></div>`;
    ["stratified-table-wrap", "uf-heatmap-wrap", "error-regime-table-wrap", "notable-weeks-grid"].forEach(
      (id) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = msgHtml;
      }
    );
    ["calibration-svg", "pit-svg", "pr-svg"].forEach((id) => {
      const svg = document.getElementById(id);
      if (svg) {
        svg.innerHTML = `<text x="230" y="140" text-anchor="middle" font-size="11" fill="#898781">Diagnósticos indisponíveis no momento.</text>`;
      }
    });
    document.querySelectorAll<HTMLButtonElement>(".js-retry-diag").forEach((btn) => {
      btn.addEventListener("click", () => {
        void this.loadDiagnostics();
      });
    });
  }

  private async loadDiagnostics(): Promise<void> {
    try {
      const res = await fetch("./data/diagnostics.json");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.diagnostics = (await res.json()) as DiagnosticsData;

      const pitSelect = document.getElementById("pit-model-select") as HTMLSelectElement | null;
      if (pitSelect && this.diagnostics.pit.models.length > 0) {
        if (!this.diagnostics.pit.models.some((m) => m.model === this.pitModel)) {
          this.pitModel = this.diagnostics.pit.models[0].model;
        }
        pitSelect.innerHTML = this.diagnostics.pit.models
          .map(
            (m) =>
              `<option value="${escapeHtml(m.model)}" ${m.model === this.pitModel ? "selected" : ""}>${escapeHtml(SHORT_LABELS[m.model] || m.label)}</option>`
          )
          .join("");
      }

      const ufSelect = document.getElementById("uf-model-select") as HTMLSelectElement | null;
      if (ufSelect && this.diagnostics.uf_skill.models.length > 0) {
        if (!this.diagnostics.uf_skill.models.some((m) => m.model === this.ufModel)) {
          this.ufModel = this.diagnostics.uf_skill.models[0].model;
        }
        ufSelect.innerHTML = this.diagnostics.uf_skill.models
          .map(
            (m) =>
              `<option value="${escapeHtml(m.model)}" ${m.model === this.ufModel ? "selected" : ""}>${escapeHtml(SHORT_LABELS[m.model] || m.label)}</option>`
          )
          .join("");
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
      this.renderDiagnosticsError();
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
      .map((s) => `<th>${escapeHtml(formatStratumName(this.stratumDim, s))}</th>`)
      .join("");

    const bodyRows = models
      .map((m) => {
        const sample = rows.find((r) => r.model === m);
        const label = escapeHtml(sample?.label || m);
        const isServed = m === "lgbm_cqr_level";
        const cells = strata
          .map((s) => {
            const rec = rows.find((r) => r.model === m && r.stratum === s);
            if (!rec || rec.rel_wis == null) return "<td>—</td>";
            const bg = skillBgColor(rec.rel_wis);
            const covTxt =
              rec.cov90 != null
                ? ` <small style="color:#52514e">(${formatPercent(rec.cov90, 0)})</small>`
                : "";
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
    const out = buildCalibrationSvg(this.diagnostics, this.calMode);
    svg.innerHTML = out.svg;
    leg.innerHTML = out.legendHtml;
  }

  private renderPitChart(): void {
    const svg = document.getElementById("pit-svg");
    const cap = document.getElementById("pit-caption");
    if (!svg || !this.diagnostics) return;
    const out = buildPitSvg(this.diagnostics, this.pitModel, this.pitSlice);
    svg.innerHTML = out.svg;
    if (cap) cap.textContent = out.caption;
  }

  private renderPrChart(): void {
    const svg = document.getElementById("pr-svg");
    const leg = document.getElementById("pr-legend");
    if (!svg || !leg || !this.diagnostics) return;
    const out = buildPrSvg(this.diagnostics, this.prHorizon);
    svg.innerHTML = out.svg;
    leg.innerHTML = out.legendHtml;
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
        return `<tr><td><strong>${escapeHtml(u.uf)}</strong></td><td>${escapeHtml(u.region)}</td>${cells}</tr>`;
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
            <td><strong>${escapeHtml(r.label)}</strong></td>
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
            <th>Viés Médio (obs − q50)</th>
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
              <span class="notable-week-badge">SE ${escapeHtml(formatEpiweek(w.week))}</span>
              <span class="notable-week-stats">Acima de q95: <strong>${formatPercent(w.under90_share, 1)}</strong></span>
            </div>
            <p class="card-sub" style="margin-top:6px">${escapeHtml(w.description)}</p>
            <p class="notable-week-stats" style="margin-top:6px">
              Total Brasil: <strong>${w.obs_total.toLocaleString("pt-BR")}</strong> obs vs
              <strong>${w.pred_total.toLocaleString("pt-BR")}</strong> soma das medianas (h=8)
            </p>
          </div>
          <button type="button" class="btn-jump-map" data-jump-week="${w.week}">Abrir no mapa de resíduos →</button>
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
