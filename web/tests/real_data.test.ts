import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildCalibrationSvg,
  buildOriginConeSvg,
  buildPitSvg,
  buildPrSvg,
  buildRollingSeriesSvg,
} from "../src/lib/charts";
import { computeDrawerKpis, decodeMapByte, searchMunicipalities } from "../src/lib/formatters";
import type { DiagnosticsData, FanData, LiveData, MapMeta, Municipality } from "../src/types";

const DATA_DIR = resolve(process.cwd(), "public/data");
const HAS_EXPORTED_DATA = existsSync(resolve(DATA_DIR, "map/meta.json"));

describe.skipIf(!HAS_EXPORTED_DATA)("Real exported data end-to-end compatibility", () => {
  it("parses and renders real meta, municipalities, live, diagnostics, binaries, and fan charts across boundary weeks", () => {
    const meta: MapMeta = JSON.parse(readFileSync(resolve(DATA_DIR, "map/meta.json"), "utf-8"));
    const munis: Municipality[] = JSON.parse(
      readFileSync(resolve(DATA_DIR, "municipalities.json"), "utf-8")
    );
    const diag: DiagnosticsData = JSON.parse(
      readFileSync(resolve(DATA_DIR, "diagnostics.json"), "utf-8")
    );
    const live: LiveData = JSON.parse(readFileSync(resolve(DATA_DIR, "live.json"), "utf-8"));

    expect(meta.weeks.length).toBeGreaterThan(200);
    expect(munis.length).toBe(meta.geocodes.length);
    expect(live.geocodes.length).toBe(meta.geocodes.length);

    // Search over all 5,570 real Brazilian municipalities
    expect(searchMunicipalities(munis, "sao paulo")[0].geocode).toBe(3550308);
    expect(searchMunicipalities(munis, "sorocaba sp")[0].geocode).toBe(3552205);

    // Binary decoding over real map layers
    const obsBuf = new Uint8Array(readFileSync(resolve(DATA_DIR, "map/obs.bin")));
    const err8Buf = new Uint8Array(readFileSync(resolve(DATA_DIR, "map/err_h8.bin")));
    const p8Buf = new Uint8Array(readFileSync(resolve(DATA_DIR, "map/p_h8.bin")));
    expect(obsBuf.length).toBe(meta.weeks.length * meta.geocodes.length);
    expect(err8Buf.length).toBe(obsBuf.length);
    expect(p8Buf.length).toBe(obsBuf.length);

    for (let g = 0; g < 50; g++) {
      const vObs = decodeMapByte(obsBuf[g], "cases", meta.inc_scale);
      const vErr = decodeMapByte(err8Buf[g], "residual", meta.inc_scale, meta.res_scale, meta.res_zero);
      const vP = decodeMapByte(p8Buf[g], "outbreak");
      expect(vObs === null || Number.isFinite(vObs)).toBe(true);
      expect(vErr === null || Number.isFinite(vErr)).toBe(true);
      expect(vP === null || (vP >= 0 && vP <= 100)).toBe(true);
    }

    // Representative municipalities: metropolis, medium-large (Sorocaba), capital, Amazonian, tiny <5k
    const tinyMuni = munis.find((m) => m.population < 3000) || munis[0];
    const sampleCodes = [3550308, 3552205, 5300108, 1302603, tinyMuni.geocode];

    for (const code of sampleCodes) {
      const fan: FanData = JSON.parse(
        readFileSync(resolve(DATA_DIR, `fan/${code}.json`), "utf-8")
      );
      const gIdx = meta.geocodes.indexOf(code);
      expect(gIdx).toBeGreaterThanOrEqual(0);

      for (const wIdx of [0, Math.floor(meta.weeks.length / 2), meta.weeks.length - 1]) {
        const wk = meta.weeks[wIdx];
        for (const h of [1, 2, 4, 8]) {
          const s1 = buildRollingSeriesSvg(fan, h, wk);
          expect(s1.length).toBeGreaterThan(100);
          expect(s1).not.toContain("NaN");
          expect(s1).not.toContain("undefined");
        }
        const s2 = buildOriginConeSvg(fan, wk);
        expect(s2.length).toBeGreaterThan(100);
        expect(s2).not.toContain("NaN");
        expect(s2).not.toContain("undefined");

        const snap = computeDrawerKpis({
          fanData: fan,
          meta,
          weekIndex: wIdx,
          horizon: 8,
          pCode: p8Buf[wIdx * meta.geocodes.length + gIdx],
        });
        expect(snap.casesText).not.toBe("—");
        expect(snap.predText).not.toBe("—");
      }
    }

    // All diagnostic SVG charts over real diagnostics.json
    for (const mode of ["overall", "weekly2024"] as const) {
      const cal = buildCalibrationSvg(diag, mode);
      expect(cal.svg).not.toContain("NaN");
    }
    for (const m of diag.pit.models) {
      for (const sl of ["h1", "h4", "h8", "h8_2024", "h8_active"]) {
        const pit = buildPitSvg(diag, m.model, sl);
        expect(pit.svg).not.toContain("NaN");
      }
    }
    for (const h of [1, 2, 4, 8]) {
      const pr = buildPrSvg(diag, h);
      expect(pr.svg).not.toContain("NaN");
    }
  });
});
