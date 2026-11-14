import maplibregl from "maplibre-gl";

interface Municipality {
  geocode: number;
  name: string;
  uf: string;
  region: string;
  population?: number;
}

interface BenchmarkRow {
  model: string;
  h1: string;
  h2: string;
  h4: string;
  h8: string;
  cov90: string;
  pr_auc: string;
}

const DEFAULT_BENCHMARKS: BenchmarkRow[] = [
  {
    model: "Ingênuo por nível (baseline)",
    h1: "2,10 (1,00×)",
    h2: "3,17 (1,00×)",
    h4: "5,15 (1,00×)",
    h8: "8,61 (1,00×)",
    cov90: "92,5%",
    pr_auc: "0,234",
  },
  {
    model: "LightGBM Global (quantílico)",
    h1: "1,80 (0,86×)",
    h2: "2,53 (0,80×)",
    h4: "3,84 (0,75×)",
    h8: "6,43 (0,75×)",
    cov90: "92,0%",
    pr_auc: "0,434",
  },
  {
    model: "LightGBM + CQR Estático",
    h1: "1,78 (0,85×)",
    h2: "2,50 (0,79×)",
    h4: "3,80 (0,74×)",
    h8: "6,38 (0,74×)",
    cov90: "90,4%",
    pr_auc: "0,436",
  },
  {
    model: "LightGBM + ACI Adaptativo",
    h1: "1,77 (0,84×)",
    h2: "2,48 (0,78×)",
    h4: "3,79 (0,74×)",
    h8: "6,35 (0,74×)",
    cov90: "90,1%",
    pr_auc: "0,441",
  },
];

const SAMPLE_MUNICIPALITIES: Municipality[] = [
  { geocode: 3550308, name: "São Paulo", uf: "SP", region: "Sudeste", population: 11451245 },
  { geocode: 3304557, name: "Rio de Janeiro", uf: "RJ", region: "Sudeste", population: 6211423 },
  { geocode: 3106200, name: "Belo Horizonte", uf: "MG", region: "Sudeste", population: 2315560 },
  { geocode: 5300108, name: "Brasília", uf: "DF", region: "Centro-Oeste", population: 2817068 },
  { geocode: 4106902, name: "Curitiba", uf: "PR", region: "Sul", population: 1773733 },
  { geocode: 2927408, name: "Salvador", uf: "BA", region: "Nordeste", population: 2418005 },
  { geocode: 1302603, name: "Manaus", uf: "AM", region: "Norte", population: 2063547 },
];

class App {
  private map: maplibregl.Map | null = null;
  private isPlaying: boolean = false;
  private currentWeekIndex: number = 0;
  private playTimer: number | null = null;
  private activeMetric: "cases" | "q50" | "outbreak" = "cases";
  private activeHorizon: number = 1;
  private municipalities: Municipality[] = SAMPLE_MUNICIPALITIES;

  constructor() {
    this.initTabs();
    this.initMap();
    this.initControls();
    this.initSearch();
    this.renderBenchmarkTable();
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

  private initMap(): void {
    const container = document.getElementById("map-container");
    if (!container) return;

    this.map = new maplibregl.Map({
      container,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: [
              "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
              "https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
            ],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors, © CARTO",
          },
        },
        layers: [
          {
            id: "osm-layer",
            type: "raster",
            source: "osm",
            minzoom: 0,
            maxzoom: 18,
          },
        ],
      },
      center: [-52.0, -14.0],
      zoom: 3.8,
      minZoom: 3.0,
      maxZoom: 12.0,
    });

    this.map.addControl(new maplibregl.NavigationControl(), "top-right");

    this.map.on("load", () => {
      this.loadMapData();
    });
  }

  private async loadMapData(): Promise<void> {
    // Carrega dados de municípios se disponíveis
    try {
      const res = await fetch("./data/municipalities.json");
      if (res.ok) {
        this.municipalities = await res.json();
      }
    } catch {
      // Usa amostra padrão
    }
  }

  private initControls(): void {
    // Métrica
    const metricBtns = document.querySelectorAll<HTMLButtonElement>(".btn-toggle");
    metricBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        metricBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.activeMetric = (btn.getAttribute("data-metric") as any) || "cases";
        this.updateLegend();
      });
    });

    // Horizonte
    const horizonBtns = document.querySelectorAll<HTMLButtonElement>(".btn-pill");
    horizonBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        horizonBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.activeHorizon = parseInt(btn.getAttribute("data-horizon") || "1", 10);
      });
    });

    // Timeline Slider
    const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
    const weekLabel = document.getElementById("current-epiweek");
    const playBtn = document.getElementById("btn-play-timeline");

    if (slider) {
      slider.addEventListener("input", () => {
        this.currentWeekIndex = parseInt(slider.value, 10);
        if (weekLabel) {
          const w = (this.currentWeekIndex + 1).toString().padStart(2, "0");
          weekLabel.textContent = `2024-${w}`;
        }
      });
    }

    if (playBtn) {
      playBtn.addEventListener("click", () => {
        this.togglePlay();
      });
    }

    // Drawer Close
    const closeBtn = document.getElementById("btn-close-drawer");
    const drawer = document.getElementById("municipality-drawer");
    if (closeBtn && drawer) {
      closeBtn.addEventListener("click", () => {
        drawer.classList.remove("open");
      });
    }
  }

  private togglePlay(): void {
    const playBtn = document.getElementById("btn-play-timeline");
    const slider = document.getElementById("timeline-range") as HTMLInputElement | null;
    const weekLabel = document.getElementById("current-epiweek");

    this.isPlaying = !this.isPlaying;
    if (playBtn) {
      playBtn.textContent = this.isPlaying ? "❚❚" : "▶";
    }

    if (this.isPlaying) {
      this.playTimer = window.setInterval(() => {
        this.currentWeekIndex = (this.currentWeekIndex + 1) % 52;
        if (slider) slider.value = this.currentWeekIndex.toString();
        if (weekLabel) {
          const w = (this.currentWeekIndex + 1).toString().padStart(2, "0");
          weekLabel.textContent = `2024-${w}`;
        }
      }, 500);
    } else {
      if (this.playTimer) {
        clearInterval(this.playTimer);
        this.playTimer = null;
      }
    }
  }

  private updateLegend(): void {
    const title = document.getElementById("legend-title");
    if (!title) return;

    if (this.activeMetric === "outbreak") {
      title.textContent = "Probabilidade de Surto (>300/100k)";
    } else if (this.activeMetric === "q50") {
      title.textContent = `Previsão Mediana h=${this.activeHorizon} (incidência/100k)`;
    } else {
      title.textContent = "Casos Observados (incidência/100k)";
    }
  }

  private initSearch(): void {
    const input = document.getElementById("search-input") as HTMLInputElement | null;
    const results = document.getElementById("search-results");
    if (!input || !results) return;

    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      if (q.length < 2) {
        results.classList.add("hidden");
        results.innerHTML = "";
        return;
      }

      const matches = this.municipalities
        .filter((m) => m.name.toLowerCase().includes(q) || m.geocode.toString().startsWith(q))
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

  public openMunicipalityDrawer(muni: Municipality): void {
    const drawer = document.getElementById("municipality-drawer");
    const nameEl = document.getElementById("drawer-muni-name");
    const metaEl = document.getElementById("drawer-muni-meta");
    const casesEl = document.getElementById("drawer-kpi-cases");
    const predEl = document.getElementById("drawer-kpi-pred");

    if (nameEl) nameEl.textContent = `${muni.name} — ${muni.uf}`;
    if (metaEl) {
      const popStr = muni.population ? `Pop: ${(muni.population / 1e6).toFixed(1)}M` : `Região: ${muni.region}`;
      metaEl.textContent = `Geocode: ${muni.geocode} | ${popStr}`;
    }

    if (casesEl) casesEl.textContent = "1.840";
    if (predEl) predEl.textContent = "2.420";

    this.renderFanChart(muni);
    drawer?.classList.add("open");
  }

  private renderFanChart(_muni: Municipality): void {
    const svg = document.getElementById("fanchart-svg") as SVGSVGElement | null;
    if (!svg) return;

    svg.innerHTML = "";
    const w = 500;
    const h = 240;
    const pad = { top: 20, right: 30, bottom: 30, left: 50 };

    // Eixos
    const innerW = w - pad.left - pad.right;
    const innerH = h - pad.top - pad.bottom;

    // Dados simulados realistas: semanas 1 a 12 histórico, 13 a 20 previsão
    const weeksHist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const casesHist = [200, 240, 310, 480, 750, 1100, 1550, 2100, 2800, 3400, 3900, 4100];

    const weeksPred = [12, 13, 14, 16, 20];
    const q50 = [4100, 4300, 4450, 4200, 3100];
    const q05 = [4100, 3600, 3200, 2500, 1500];
    const q95 = [4100, 5200, 6100, 6800, 5900];
    const q25 = [4100, 4000, 3900, 3400, 2300];
    const q75 = [4100, 4700, 5100, 5200, 4200];

    const maxVal = 7500;
    const minW = 1;
    const maxW = 20;

    const scaleX = (val: number) => pad.left + ((val - minW) / (maxW - minW)) * innerW;
    const scaleY = (val: number) => pad.top + innerH - (val / maxVal) * innerH;

    // Linha divisória entre passado e futuro
    const splitX = scaleX(12);
    const splitLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    splitLine.setAttribute("x1", splitX.toString());
    splitLine.setAttribute("y1", pad.top.toString());
    splitLine.setAttribute("x2", splitX.toString());
    splitLine.setAttribute("y2", (pad.top + innerH).toString());
    splitLine.setAttribute("stroke", "#c3c2b7");
    splitLine.setAttribute("stroke-dasharray", "3,3");
    svg.appendChild(splitLine);

    // Fan envelopes (90% e 50%)
    const makePolygon = (wArr: number[], loArr: number[], hiArr: number[], fill: string) => {
      const ptsLo = wArr.map((wk, i) => `${scaleX(wk)},${scaleY(loArr[i])}`);
      const ptsHi = wArr.map((wk, i) => `${scaleX(wk)},${scaleY(hiArr[i])}`).reverse();
      const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      polygon.setAttribute("points", `${ptsLo.join(" ")} ${ptsHi.join(" ")}`);
      polygon.setAttribute("fill", fill);
      return polygon;
    };

    // 90% envelope
    svg.appendChild(makePolygon(weeksPred, q05, q95, "rgba(42, 120, 214, 0.15)"));
    // 50% envelope
    svg.appendChild(makePolygon(weeksPred, q25, q75, "rgba(42, 120, 214, 0.28)"));

    // Linha mediana de previsão
    const predPts = weeksPred.map((wk, i) => `${scaleX(wk)},${scaleY(q50[i])}`).join(" ");
    const predPoly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    predPoly.setAttribute("points", predPts);
    predPoly.setAttribute("fill", "none");
    predPoly.setAttribute("stroke", "#2a78d6");
    predPoly.setAttribute("stroke-width", "2.2");
    predPoly.setAttribute("stroke-dasharray", "4,3");
    svg.appendChild(predPoly);

    // Linha histórica de casos observados
    const histPts = weeksHist.map((wk, i) => `${scaleX(wk)},${scaleY(casesHist[i])}`).join(" ");
    const histPoly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    histPoly.setAttribute("points", histPts);
    histPoly.setAttribute("fill", "none");
    histPoly.setAttribute("stroke", "#0b0b0b");
    histPoly.setAttribute("stroke-width", "2.0");
    svg.appendChild(histPoly);

    // Pontos observados
    weeksHist.forEach((wk, i) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", scaleX(wk).toString());
      circle.setAttribute("cy", scaleY(casesHist[i]).toString());
      circle.setAttribute("r", "3");
      circle.setAttribute("fill", "#0b0b0b");
      svg.appendChild(circle);
    });
  }

  private renderBenchmarkTable(): void {
    const tbody = document.getElementById("benchmark-table-body");
    if (!tbody) return;

    tbody.innerHTML = "";
    DEFAULT_BENCHMARKS.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.model}</strong></td>
        <td>${row.h1}</td>
        <td>${row.h2}</td>
        <td>${row.h4}</td>
        <td>${row.h8}</td>
        <td>${row.cov90}</td>
        <td>${row.pr_auc}</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

// Inicia aplicação
window.addEventListener("DOMContentLoaded", () => {
  new App();
});
