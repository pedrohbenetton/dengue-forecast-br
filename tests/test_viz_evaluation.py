import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from dengue_forecast import backtest
from dengue_forecast.baselines import Climatology, Naive, SeasonalNaive
from dengue_forecast.viz import evaluation


def test_figures_from_a_small_backtest(toy_data, tmp_path):
    logs = []
    scored = backtest.run_all(
        toy_data, [Naive(), SeasonalNaive(), Climatology()], seasons=(2022, 2023), log=logs.append
    )
    assert len(logs) == 6
    assert list(scored["model"].cat.categories) == ["naive", "seasonal", "climatology"]
    assert isinstance(evaluation.plot_skill_by_band(scored), Figure)
    assert isinstance(evaluation.plot_calibration(scored), Figure)
    paths = evaluation.make_all(scored, tmp_path)
    assert [p.name for p in paths] == ["fase1_skill_por_faixa.png", "fase1_calibracao.png"]

    tables = backtest.write_tables(scored, tmp_path / "tables")
    assert (tmp_path / "tables" / "por_faixa.csv").exists()
    assert (tables["por_horizonte"].xs("naive", level="model")["rel_wis"] == 1.0).all()
    # os dados sintéticos não têm surtos: PR-AUC indefinido, prevalência zero
    assert len(tables["alerta"]) == 12
    assert (tables["alerta"]["prevalence"] == 0).all()
    assert tables["alerta"]["pr_auc"].isna().all()
