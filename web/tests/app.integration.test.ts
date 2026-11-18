import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, type MapAdapter } from "../src/app";

const INDEX_HTML = readFileSync(resolve(process.cwd(), "index.html"), "utf-8");

function extractBodyHtml(html: string): string {
  const m = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  if (!m) throw new Error("Missing <body> in index.html");
  return m[1].replace(/<script[\s\S]*?<\/script>/gi, "");
}

class MockMap implements MapAdapter {
  public handlers: Map<string, ((ev?: any) => void)[]> = new Map();
  public layers: Map<string, any> = new Map();
  public sources: Map<string, any> = new Map();
  public featureStates: Map<string, any> = new Map();
  public paintProps: Map<string, any> = new Map();
  public filters: Map<string, any> = new Map();
  public lastBounds: [number, number, number, number] | null = null;
  public resizeCount: number = 0;
  private canvasCursor = { style: { cursor: "" } };

  on(type: string, arg2: string | ((ev?: any) => void), arg3?: (ev: any) => void): void {
    const key = typeof arg2 === "string" ? `${type}:${arg2}` : type;
    const cb = typeof arg2 === "function" ? arg2 : arg3!;
    const arr = this.handlers.get(key) || [];
    arr.push(cb);
    this.handlers.set(key, arr);
  }

  emit(key: string, ev?: any): void {
    (this.handlers.get(key) || []).forEach((fn) => fn(ev));
  }

  addSource(id: string, source: any): void {
    this.sources.set(id, source);
  }

  addLayer(layer: any): void {
    this.layers.set(layer.id, layer);
  }

  getLayer(id: string): any {
    return this.layers.get(id);
  }

  setPaintProperty(layer: string, name: string, value: any): void {
    this.paintProps.set(`${layer}:${name}`, value);
  }

  setFilter(layer: string, filter: any): void {
    this.filters.set(layer, filter);
  }

  setFeatureState(target: { source: string; id: string }, state: Record<string, any>): void {
    this.featureStates.set(target.id, state.v);
  }

  fitBounds(bounds: [number, number, number, number]): void {
    this.lastBounds = bounds;
  }

  getCanvas() {
    return this.canvasCursor;
  }

  resize(): void {
    this.resizeCount++;
  }
}

function makeMiniTopo() {
  return {
    type: "Topology",
    objects: {
      BRMU: {
        type: "GeometryCollection",
        geometries: [
          {
            type: "Polygon",
            properties: { codarea: "3550308" },
            arcs: [[0]],
          },
          {
            type: "Polygon",
            properties: { codarea: "3552205" },
            arcs: [[0]],
          },
        ],
      },
    },
    arcs: [
      [
        [-47.0, -23.0],
        [-46.0, -23.0],
        [-46.0, -24.0],
        [-47.0, -24.0],
        [-47.0, -23.0],
      ],
    ],
  };
}

const MUNIS = [
  { geocode: 3550308, name: "São Paulo", uf: "SP", region: "Sudeste", population: 11451999 },
  { geocode: 3552205, name: "Sorocaba", uf: "SP", region: "Sudeste", population: 738128 },
];

const META = {
  weeks: [202408, 202412, 202640],
  geocodes: [3550308, 3552205],
  horizons: [1, 2, 4, 8],
  inc_scale: 32,
  res_scale: 32,
  res_zero: 127,
  latest_week: 202640,
};

const SP_FAN = {
  weeks: [202408, 202412, 202640],
  obs: [32000, 75293, 210],
  h1: {
    "q0.05": [25000, 50000, 150],
    "q0.25": [28000, 55000, 180],
    "q0.5": [30000, 60782.1, 200],
    "q0.75": [34000, 66000, 230],
    "q0.95": [40000, 78000, 280],
  },
  h8: {
    "q0.05": [10000, 8000, 90],
    "q0.25": [12000, 12000, 130],
    "q0.5": [15000, 15160.4, 175],
    "q0.75": [19000, 21000, 220],
    "q0.95": [26000, 32000, 310],
  },
};

const SOROCABA_FAN = {
  weeks: [202408, 202412, 202640],
  obs: [1800, 4474, 95],
  h1: {
    "q0.05": [1200, 3100, 60],
    "q0.25": [1500, 3500, 78],
    "q0.5": [1700, 3861.9, 92],
    "q0.75": [1950, 4200, 110],
    "q0.95": [2400, 4900, 145],
  },
  h8: {
    "q0.05": [700, 1100, 40],
    "q0.25": [1000, 1800, 65],
    "q0.5": [1300, 2491.7, 88],
    "q0.75": [1700, 3100, 120],
    "q0.95": [2300, 4600, 185],
  },
};

const LIVE_FIXTURE = {
  origin: 202644,
  geocodes: [3550308, 3552205],
  targets: { "1": 202645, "2": 202646, "4": 202648, "8": 202652 },
  nowcast_weeks: 4,
  h1: { "q0.05": [400, 82], "q0.5": [520, 120], "q0.95": [680, 159], p: [0.025, 0.025] },
  h2: { "q0.05": [380, 75], "q0.5": [540, 128], "q0.95": [740, 185], p: [0.025, 0.025] },
  h4: { "q0.05": [350, 72], "q0.5": [580, 140], "q0.95": [820, 193], p: [0.025, 0.025] },
  h8: { "q0.05": [310, 56], "q0.5": [610, 143], "q0.95": [950, 305], p: [0.025, 0.025] },
};

function jsonResponse(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function binResponse(bytes: number[]): Response {
  return new Response(new Uint8Array(bytes).buffer, { status: 200 });
}

async function flushPromises(ticks = 6): Promise<void> {
  for (let i = 0; i < ticks; i++) {
    await Promise.resolve();
  }
}

describe("App DOM & Async State Integration", () => {
  let mockMap: MockMap;
  let app: App | null = null;

  beforeEach(() => {
    document.body.innerHTML = extractBodyHtml(INDEX_HTML);
    mockMap = new MockMap();
    Element.prototype.scrollIntoView = vi.fn();
  });

  afterEach(() => {
    app?.destroy();
    app = null;
    vi.restoreAllMocks();
  });

  it("NEVER leaks São Paulo KPIs or scrollTop onto Sorocaba on network error, and recovers via Retry", async () => {
    let failSorocaba = true;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("municipios.topo.json")) return jsonResponse(makeMiniTopo());
        if (url.endsWith("municipalities.json")) return jsonResponse(MUNIS);
        if (url.endsWith("map/meta.json")) return jsonResponse(META);
        if (url.endsWith("benchmarks.json")) return jsonResponse({ rows: [], reference: "naive_level", source: "test" });
        if (url.endsWith("diagnostics.json"))
          return jsonResponse({
            reference: "naive_level",
            stratified: {},
            uf_skill: { models: [], ufs: [] },
            calibration: { nominals: [0.5, 0.8, 0.9, 0.95], curves: [], weekly_2024: { origins: [], series: [] } },
            pit: { bins: 10, models: [] },
            pr_curves: { recall: [], models: [] },
            error_analysis: { regimes_by_horizon: {}, notable_weeks: [] },
          });
        if (url.endsWith("live.json")) return jsonResponse(LIVE_FIXTURE);
        if (url.endsWith("obs.bin")) return binResponse([100, 80, 200, 180, 40, 30]);
        if (url.endsWith("p_h1.bin")) return binResponse([50, 20, 220, 200, 6, 6]);
        if (url.endsWith("fan/3550308.json")) return jsonResponse(SP_FAN);
        if (url.endsWith("p_h8.bin")) {
          if (failSorocaba) throw new TypeError("Failed to fetch");
          return binResponse([40, 20, 180, 190, 6, 6]);
        }
        if (url.endsWith("fan/3552205.json")) {
          if (failSorocaba) throw new TypeError("Failed to fetch");
          return jsonResponse(SOROCABA_FAN);
        }
        return new Response("Not found", { status: 404 });
      })
    );

    app = new App({ mapFactory: () => mockMap });
    mockMap.emit("load");
    await flushPromises(10);

    // 1. Open São Paulo at 2024-12 (h=1)
    await app.openMunicipalityDrawer(MUNIS[0]);
    await flushPromises(8);

    expect(document.getElementById("drawer-muni-name")?.textContent).toBe("São Paulo — SP");
    expect(document.getElementById("drawer-kpi-cases")?.textContent).toBe("75.293");
    expect(document.getElementById("drawer-kpi-pred")?.textContent).toBe("60.782");
    expect(document.getElementById("drawer-kpi-res")?.textContent).toBe("+14.511");

    // Simulate user scrolling down in drawer body on São Paulo
    const drawerBody = document.querySelector<HTMLElement>(".drawer-body")!;
    drawerBody.scrollTop = 145;

    // 2. Switch horizon to h=8 and click Sorocaba while network fails (exact screenshot scenario!)
    document.querySelector<HTMLButtonElement>('.btn-pill[data-horizon="8"]')?.click();
    await app.openMunicipalityDrawer(MUNIS[1]);
    await flushPromises(8);

    expect(document.getElementById("drawer-muni-name")?.textContent).toBe("Sorocaba — SP");
    expect(drawerBody.scrollTop).toBe(0);
    // CRITICAL ASSERTIONS: Zero stale São Paulo values!
    expect(document.getElementById("drawer-kpi-cases")?.textContent).toBe("—");
    expect(document.getElementById("drawer-kpi-pred")?.textContent).toBe("—");
    expect(document.getElementById("drawer-kpi-res")?.textContent).toBe("—");
    expect(document.getElementById("drawer-kpi-outbreak")?.textContent).toBe("—");
    expect(document.getElementById("drawer-kpi-res")?.classList.contains("pos-bias")).toBe(false);

    // Live forecast uses <= 2,5% consistently
    expect(document.getElementById("drawer-live")?.textContent).toContain("120");
    expect(document.getElementById("drawer-live")?.textContent).toContain("≤ 2,5%");

    // Error overlay + Retry button rendered
    const fanState = document.getElementById("fanchart-state")!;
    expect(fanState.classList.contains("hidden")).toBe(false);
    const retryBtn = fanState.querySelector<HTMLButtonElement>(".btn-retry")!;
    expect(retryBtn).toBeTruthy();

    // 3. Server comes back up -> clicking Retry loads Sorocaba's real h=8 KPIs & Fan Chart!
    failSorocaba = false;
    retryBtn.click();
    await flushPromises(10);

    expect(fanState.classList.contains("hidden")).toBe(true);
    expect(document.getElementById("drawer-kpi-cases")?.textContent).toBe("4.474");
    expect(document.getElementById("drawer-kpi-pred")?.textContent).toBe("2.492");
    expect(document.getElementById("drawer-kpi-res")?.textContent).toBe("+1.982");
    expect(document.getElementById("drawer-kpi-outbreak")?.textContent).toBe("75%");
    expect(document.getElementById("fanchart-svg")?.innerHTML).toContain("<path");
  });

  it("handles jumpToResidualWeek called BEFORE meta.json resolves and refreshes live.json late arrival", async () => {
    let resolveMeta!: (r: Response) => void;
    let resolveLive!: (r: Response) => void;

    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("municipios.topo.json")) return Promise.resolve(jsonResponse(makeMiniTopo()));
        if (url.endsWith("municipalities.json")) return Promise.resolve(jsonResponse(MUNIS));
        if (url.endsWith("map/meta.json")) {
          return new Promise<Response>((res) => {
            resolveMeta = res;
          });
        }
        if (url.endsWith("live.json")) {
          return new Promise<Response>((res) => {
            resolveLive = res;
          });
        }
        if (url.endsWith("err_h8.bin")) return Promise.resolve(binResponse([127, 127, 159, 140, 120, 120]));
        if (url.endsWith("obs.bin")) return Promise.resolve(binResponse([10, 10, 20, 20, 5, 5]));
        if (url.endsWith("p_h8.bin")) return Promise.resolve(binResponse([6, 6, 180, 190, 6, 6]));
        if (url.endsWith("fan/3552205.json")) return Promise.resolve(jsonResponse(SOROCABA_FAN));
        return Promise.resolve(jsonResponse({ rows: [], stratified: {}, uf_skill: { models: [], ufs: [] }, calibration: { nominals: [], curves: [], weekly_2024: { origins: [], series: [] } }, pit: { bins: 10, models: [] }, pr_curves: { recall: [], models: [] }, error_analysis: { regimes_by_horizon: {}, notable_weeks: [] } }));
      })
    );

    app = new App({ mapFactory: () => mockMap });
    mockMap.emit("load");

    // User jumps to 2024-08 BEFORE meta.json resolves!
    app.jumpToResidualWeek(202408, 8);
    // User also opens Sorocaba BEFORE live.json resolves!
    void app.openMunicipalityDrawer(MUNIS[1]);
    await flushPromises(6);

    expect(document.getElementById("drawer-live")?.textContent).toContain("Carregando previsão ao vivo");

    // Now meta.json resolves: week must be 2024-08 (index 0), NOT overwritten by default 2024-12!
    resolveMeta(jsonResponse(META));
    await flushPromises(12);
    expect(document.getElementById("current-epiweek")?.textContent).toBe("2024-08");

    // Now live.json resolves: drawer-live must auto-populate for the already-open Sorocaba drawer!
    resolveLive(jsonResponse(LIVE_FIXTURE));
    await flushPromises(8);
    expect(document.getElementById("drawer-live")?.textContent).toContain("origem 2026-44");
    expect(document.getElementById("drawer-live")?.textContent).toContain("143");
  });

  it("supports keyboard search selection, Escape drawer close, and unindexed IBGE polygon click", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("municipios.topo.json")) return jsonResponse(makeMiniTopo());
        if (url.endsWith("municipalities.json")) return jsonResponse(MUNIS);
        if (url.endsWith("map/meta.json")) return jsonResponse(META);
        if (url.endsWith("live.json")) return jsonResponse(LIVE_FIXTURE);
        if (url.endsWith("obs.bin")) return binResponse([10, 10, 20, 20, 5, 5]);
        if (url.endsWith("p_h1.bin")) return binResponse([6, 6, 180, 190, 6, 6]);
        if (url.endsWith("fan/3552205.json")) return jsonResponse(SOROCABA_FAN);
        return jsonResponse({ rows: [], stratified: {}, uf_skill: { models: [], ufs: [] }, calibration: { nominals: [], curves: [], weekly_2024: { origins: [], series: [] } }, pit: { bins: 10, models: [] }, pr_curves: { recall: [], models: [] }, error_analysis: { regimes_by_horizon: {}, notable_weeks: [] } });
      })
    );

    app = new App({ mapFactory: () => mockMap });
    mockMap.emit("load");
    await flushPromises(10);

    const searchInput = document.getElementById("search-input") as HTMLInputElement;
    searchInput.value = "Sorocaba SP";
    searchInput.dispatchEvent(new Event("input"));

    const results = document.getElementById("search-results")!;
    expect(results.classList.contains("hidden")).toBe(false);
    expect(results.textContent).toContain("Sorocaba — SP (3552205)");

    // Press Enter -> selects Sorocaba and zooms map bounds
    searchInput.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await flushPromises(8);

    const drawer = document.getElementById("municipality-drawer")!;
    expect(drawer.classList.contains("open")).toBe(true);
    expect(mockMap.lastBounds).not.toBeNull();

    // Clicking an unindexed IBGE polygon (e.g. 5101837) shows graceful message without error
    mockMap.emit("click:municipios-fill", {
      features: [{ properties: { codarea: "5101837", nome: "Boa Esperança do Norte" } }],
    });
    expect(document.getElementById("drawer-muni-name")?.textContent).toBe("Boa Esperança do Norte");
    expect(document.getElementById("fanchart-state")?.textContent).toContain("não possui série");

    // Pressing Escape closes the drawer
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(drawer.classList.contains("open")).toBe(false);
  });
});
