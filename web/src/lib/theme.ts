// Paleta única do site, espelhando src/dengue_forecast/viz/style.py (figuras do relatório)
// e os tokens de web/src/style.css. Nenhuma cor de interface fica fora daqui.

export const T = {
  surface: "#fcfcfb",
  mapBackground: "#f4f3ef",
  ink: "#0b0b0b",
  inkSecondary: "#52514e",
  inkMuted: "#6b6a65", // contraste AA sobre o papel (o #898781 das figuras não passa em texto)
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  noData: "#e9e8e2",
  outline: "#ffffff",
  accent: "#2a78d6",
  band90: "rgba(42, 120, 214, 0.16)",
  band50: "rgba(42, 120, 214, 0.32)",
  highlight: "#eb6834", // semana/origem selecionada
  danger: "#c8402c",
} as const;

// Escalas do mapa (paradas usadas tanto no MapLibre quanto nas legendas).
export const INCIDENCE_STOPS: [number, string][] = [
  [0, "#f7f4e9"],
  [10, "#fde3a7"],
  [50, "#f9b26b"],
  [100, "#ee7a47"],
  [300, "#c8402c"],
  [1000, "#7a1020"],
];

export const PROBABILITY_STOPS: [number, string][] = [
  [0, "#f2f5fb"],
  [10, "#c6d8f0"],
  [30, "#7fa8dc"],
  [60, "#3c6fc0"],
  [90, "#173a7a"],
];

export const RESIDUAL_STOPS: [number, string][] = [
  [-2.5, "#184a99"],
  [-1.0, "#82b3e8"],
  [0.0, "#f4f3ef"],
  [1.0, "#f59e56"],
  [2.2, "#c8402c"],
  [3.2, "#7a1020"],
];

export function gradient(stops: [number, string][]): string {
  return `linear-gradient(to right, ${stops.map(([, c]) => c).join(", ")})`;
}
