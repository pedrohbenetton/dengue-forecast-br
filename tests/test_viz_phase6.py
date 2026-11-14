import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from dengue_forecast.viz import phase6


def test_plot_immunity_ablation():
    ablacao, _, _ = phase6._default_tables()
    fig = phase6.plot_immunity_ablation(ablacao)
    assert isinstance(fig, Figure)


def test_plot_season_shock_comparison():
    _, por_temporada, _ = phase6._default_tables()
    fig = phase6.plot_season_shock_comparison(por_temporada)
    assert isinstance(fig, Figure)


def test_plot_immunity_importance():
    _, _, importance = phase6._default_tables()
    fig = phase6.plot_immunity_importance(importance)
    assert isinstance(fig, Figure)


def test_make_all_and_main(tmp_path):
    ablacao, por_temporada, importance = phase6._default_tables()
    out_dir = tmp_path / "figures"
    paths = phase6.make_all(ablacao, por_temporada, importance, out_dir=out_dir)
    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0

    # Testa CLI main() com defaults
    rc = phase6.main(["--out-dir", str(tmp_path / "cli_figures")])
    assert rc == 0
    assert (tmp_path / "cli_figures" / "fase6_ablacao.png").exists()
