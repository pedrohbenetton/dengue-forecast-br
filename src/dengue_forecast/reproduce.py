"""Reprodução de ponta a ponta do projeto em um único comando (SPEC.md).

Executa em sequência determinística: ingestão (com cache), consolidação do painel, EDA,
backtest dos baselines (fase 1), LightGBM + ablação (fase 2), calibração conformal CQR/ACI
(fase 3), ST-GNN quando PyTorch está instalado (fase 4), ablação DENV-3 (fase 6), exportação
completa de dados/diagnósticos para o site (fase 5) e geração de todas as figuras estáticas.

Uso:
    uv run python -m dengue_forecast.reproduce
    uv run --group gnn python -m dengue_forecast.reproduce --skip-download
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from dengue_forecast import backtest, conformal, download, eda, export, panel, pipeline
from dengue_forecast.gnn import TORCH_AVAILABLE
from dengue_forecast.viz import diagnostics, evaluation, phase2, phase3, phase4, phase6


def build_plan(
    raw_dir: str = "data/raw",
    panel_path: str = "data/processed/panel.parquet",
    forecast_dir: str = "data/forecasts",
    reports_dir: str = "reports",
    web_data_dir: str = "web/public/data",
    skip_download: bool = False,
    include_gnn: bool = TORCH_AVAILABLE,
    include_site: bool = True,
    force: bool = False,
) -> list[tuple[str, Callable[[list[str]], int], list[str]]]:
    """Monta a lista ordenada de etapas (nome, função main, argv) do pipeline completo."""
    f_flag = ["--force"] if force else []
    processed_dir = str(Path(panel_path).parent)
    figures_dir = str(Path(reports_dir) / "figures")
    mesh_path = str(Path(raw_dir) / "ibge" / "malha_municipios.topo.json")
    muni_path = str(Path(raw_dir) / "ibge" / "municipios.json")
    val_dir = str(Path(forecast_dir) / "val")
    ab_dir = str(Path(forecast_dir) / "ablation")
    margins_path = str(Path(reports_dir) / "fase3" / "margens_cqr_nivel.csv")

    steps: list[tuple[str, Callable[[list[str]], int], list[str]]] = []

    if not skip_download:
        steps.append(("fase0:download", download.main, ["--raw-dir", raw_dir]))
    if not Path(panel_path).exists() or force or not skip_download:
        steps.append(
            ("fase0:panel", panel.main, ["--raw-dir", raw_dir, "--out-dir", processed_dir])
        )
    steps.append(("fase0:eda", eda.main, ["--panel", panel_path, "--out-dir", figures_dir]))

    # Fase 1: Baselines
    steps.append(
        (
            "fase1:backtest",
            backtest.main,
            [
                "--panel",
                panel_path,
                "--mesh",
                mesh_path,
                "--models",
                "naive,seasonal,climatology",
                "--forecast-dir",
                forecast_dir,
                "--out-dir",
                str(Path(reports_dir) / "fase1"),
                *f_flag,
            ],
        )
    )
    steps.append(
        (
            "fase1:figuras",
            evaluation.main,
            ["--forecast-dir", forecast_dir, "--out-dir", figures_dir],
        )
    )

    # Fase 2: Validação 2022 (para CQR), LightGBM principal e ablação
    steps.append(
        (
            "fase2:val2022",
            backtest.main,
            [
                "--panel",
                panel_path,
                "--mesh",
                mesh_path,
                "--models",
                "lgbm",
                "--seasons",
                "2022",
                "--forecast-dir",
                val_dir,
                *f_flag,
            ],
        )
    )
    steps.append(
        (
            "fase2:backtest",
            backtest.main,
            [
                "--panel",
                panel_path,
                "--mesh",
                mesh_path,
                "--models",
                "naive,naive_level,seasonal,climatology,lgbm",
                "--forecast-dir",
                forecast_dir,
                "--out-dir",
                str(Path(reports_dir) / "fase2"),
                "--reference",
                "naive_level",
                *f_flag,
            ],
        )
    )
    steps.append(
        (
            "fase2:ablacao",
            backtest.main,
            [
                "--panel",
                panel_path,
                "--mesh",
                mesh_path,
                "--models",
                "lgbm_own,lgbm_season,lgbm_nb",
                "--horizons",
                "1,8",
                "--forecast-dir",
                ab_dir,
                *f_flag,
            ],
        )
    )
    steps.append(
        (
            "fase2:figuras",
            phase2.main,
            [
                "--forecast-dir",
                forecast_dir,
                "--municipalities",
                muni_path,
                "--out-dir",
                figures_dir,
            ],
        )
    )

    # Fase 3: Calibração Conformal (CQR e ACI) + margens de produção
    steps.append(
        (
            "fase3:conformal",
            conformal.main,
            [
                "--panel",
                panel_path,
                "--forecast-dir",
                forecast_dir,
                "--val-dir",
                val_dir,
                "--out-dir",
                str(Path(reports_dir) / "fase3"),
                "--reference",
                "naive_level",
            ],
        )
    )
    steps.append(
        (
            "fase3:margens",
            pipeline.main,
            [
                "--panel",
                panel_path,
                "--forecast-dir",
                forecast_dir,
                "--margins",
                margins_path,
                "--write-margins",
            ],
        )
    )
    steps.append(
        (
            "fase3:figuras",
            phase3.main,
            ["--forecast-dir", forecast_dir, "--out-dir", figures_dir],
        )
    )

    # Fase 4: Rede em Grafo Espaço-Temporal e controle sem grafo
    if include_gnn:
        steps.append(
            (
                "fase4:backtest",
                backtest.main,
                [
                    "--panel",
                    panel_path,
                    "--mesh",
                    mesh_path,
                    "--models",
                    "naive,naive_level,lgbm,gnn,gnn_nograph",
                    "--forecast-dir",
                    forecast_dir,
                    "--out-dir",
                    str(Path(reports_dir) / "fase4"),
                    *f_flag,
                ],
            )
        )
        steps.append(
            (
                "fase4:figuras",
                phase4.main,
                ["--forecast-dir", forecast_dir, "--out-dir", figures_dir],
            )
        )

    # Fase 6: Dívida Imunológica ao DENV-3
    steps.append(
        (
            "fase6:backtest",
            backtest.main,
            [
                "--panel",
                panel_path,
                "--mesh",
                mesh_path,
                "--models",
                "lgbm_immunity",
                "--forecast-dir",
                forecast_dir,
                *f_flag,
            ],
        )
    )
    steps.append(
        (
            "fase6:figuras",
            phase6.main,
            [
                "--forecast-dir",
                forecast_dir,
                "--fase6-dir",
                str(Path(reports_dir) / "fase6"),
                "--out-dir",
                figures_dir,
            ],
        )
    )

    # Fase 5: Exportação do site estático, figuras diagnósticas e previsão ao vivo
    if include_site:
        steps.append(
            (
                "fase5:export",
                export.main,
                [
                    "--panel",
                    panel_path,
                    "--forecast-dir",
                    forecast_dir,
                    "--names",
                    muni_path,
                    "--mesh",
                    mesh_path,
                    "--reports-dir",
                    reports_dir,
                    "--out-dir",
                    web_data_dir,
                ],
            )
        )
        steps.append(
            (
                "fase5:diagnosticos",
                diagnostics.main,
                [
                    "--diagnostics",
                    str(Path(web_data_dir) / "diagnostics.json"),
                    "--out-dir",
                    figures_dir,
                ],
            )
        )
        steps.append(
            (
                "fase5:live",
                pipeline.main,
                [
                    "--panel",
                    panel_path,
                    "--mesh",
                    mesh_path,
                    "--margins",
                    margins_path,
                    "--out",
                    str(Path(web_data_dir) / "live.json"),
                ],
            )
        )

    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--forecast-dir", default="data/forecasts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--web-data-dir", default="web/public/data")
    parser.add_argument(
        "--skip-download", action="store_true", help="usa o cache atual de data/raw"
    )
    parser.add_argument("--skip-gnn", action="store_true", help="pula a fase 4 (PyTorch)")
    parser.add_argument("--skip-site", action="store_true", help="pula exportação da camada web")
    parser.add_argument("--force", action="store_true", help="recalcula previsões já existentes")
    parser.add_argument("--dry-run", action="store_true", help="apenas lista as etapas planejadas")
    args = parser.parse_args(argv)

    plan = build_plan(
        raw_dir=args.raw_dir,
        panel_path=args.panel,
        forecast_dir=args.forecast_dir,
        reports_dir=args.reports_dir,
        web_data_dir=args.web_data_dir,
        skip_download=args.skip_download,
        include_gnn=TORCH_AVAILABLE and not args.skip_gnn,
        include_site=not args.skip_site,
        force=args.force,
    )

    for idx, (name, fn, step_args) in enumerate(plan, start=1):
        print(f"[{idx}/{len(plan)}] {name}: {' '.join(step_args)}")
        if not args.dry_run:
            rc = fn(step_args)
            if rc not in (0, None):
                return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
