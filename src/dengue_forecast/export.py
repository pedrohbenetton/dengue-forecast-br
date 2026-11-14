"""Exportação de dados e métricas para a aplicação web estática (GitHub Pages).

Gera arquivos JSON compactos e otimizados para o frontend (mapa coroplético,
fan charts municipais e painel comparativo de avaliação).

Uso:
    uv run python -m dengue_forecast.export
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from dengue_forecast import backtest


def export_evaluation_summary(
    reports_dir: Path | str, out_file: Path | str
) -> dict:
    """Consolida tabelas de avaliação em um arquivo JSON unificado."""
    reports_p = Path(reports_dir)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    summary: dict = {"por_horizonte": {}, "por_temporada": {}, "por_faixa": {}, "alerta": {}}

    p_horiz = reports_p / "por_horizonte.csv"
    if p_horiz.exists():
        df = pd.read_csv(p_horiz)
        summary["por_horizonte"] = df.to_dict(orient="records")

    p_temp = reports_p / "por_temporada.csv"
    if p_temp.exists():
        df = pd.read_csv(p_temp)
        summary["por_temporada"] = df.to_dict(orient="records")

    p_faixa = reports_p / "por_faixa.csv"
    if p_faixa.exists():
        df = pd.read_csv(p_faixa)
        summary["por_faixa"] = df.to_dict(orient="records")

    p_alerta = reports_p / "alerta.csv"
    if p_alerta.exists():
        df = pd.read_csv(p_alerta)
        summary["alerta"] = df.to_dict(orient="records")

    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def export_municipalities_index(
    municipalities_df: pd.DataFrame, out_file: Path | str
) -> list[dict]:
    """Exporta índice leve de municípios para busca e navegação no mapa."""
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cols = ["geocode", "name", "uf", "region"]
    available_cols = [c for c in cols if c in municipalities_df.columns]
    records = municipalities_df[available_cols].to_dict(orient="records")

    out_p.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def export_map_timeline(
    forecasts: pd.DataFrame, out_dir: Path | str
) -> list[Path]:
    """Exporta fatias semanais compactas {geocode: [y, q50, p_outbreak]} para o mapa animado.

    Cada arquivo representa uma semana-alvo ou semana de origem, permitindo carregamento
    sob demanda rápido conforme o usuário movimenta o slider temporal.
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    paths = []
    # Agrupa por semana de origem
    for origin_week, group in forecasts.groupby("origin", observed=True):
        week_data: dict[str, dict] = {}
        for _, row in group.iterrows():
            geo_str = str(int(row["geocode"]))
            h = int(row.get("horizon", 1))
            if geo_str not in week_data:
                week_data[geo_str] = {
                    "y": float(row.get("y", 0.0)),
                    "forecasts": {},
                }
            # Adiciona dados por horizonte
            q_cols = [c for c in backtest.QCOLS if c in row]
            q_vals = [float(row[c]) for c in q_cols] if q_cols else []
            q50 = float(row.get("q0.5", row.get("y", 0.0)))
            p_outbreak = float(row.get("p_outbreak", 0.0))

            week_data[geo_str]["forecasts"][str(h)] = {
                "q50": q50,
                "p_outbreak": p_outbreak,
                "quantiles": q_vals,
            }

        target_file = out_p / f"week_{origin_week}.json"
        target_file.write_text(json.dumps(week_data, ensure_ascii=False), encoding="utf-8")
        paths.append(target_file)

    return paths


def export_fan_chart_series(
    forecasts: pd.DataFrame, out_dir: Path | str, geocodes: list[int] | None = None
) -> list[Path]:
    """Exporta a série temporal de histórico observado e leques de previsão por município."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if geocodes is None:
        geocodes = list(forecasts["geocode"].unique()[:20])

    paths = []
    for geo in geocodes:
        sub = forecasts[forecasts["geocode"] == geo].sort_values(["origin", "horizon"])
        if sub.empty:
            continue

        records = []
        for _, row in sub.iterrows():
            q_dict = {q_col: float(row[q_col]) for q_col in backtest.QCOLS if q_col in row}
            records.append(
                {
                    "origin": int(row["origin"]),
                    "target": int(row["target"]),
                    "horizon": int(row["horizon"]),
                    "y": float(row.get("y", 0.0)),
                    **q_dict,
                }
            )

        geo_file = out_p / f"geo_{geo}.json"
        geo_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        paths.append(geo_file)

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", default="reports/fase2")
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--out-dir", default="web/public/data")
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args(argv)

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Exportando dados para {out_base}...")
    export_evaluation_summary(args.reports_dir, out_base / "evaluation_summary.json")

    # Se houver municípios em cache ou IBGE sample
    sample_ibge = Path("tests/fixtures/ibge_municipios_sample.json")
    if sample_ibge.exists():
        from dengue_forecast.ibge import parse_municipalities

        df_muni = parse_municipalities(sample_ibge.read_text(encoding="utf-8"))
        export_municipalities_index(df_muni, out_base / "municipalities.json")

    print("Exportação concluída com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
