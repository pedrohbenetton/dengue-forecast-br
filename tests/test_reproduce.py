from dengue_forecast import reproduce


def test_reproduce_plan_and_dry_run(capsys):
    plan = reproduce.build_plan(skip_download=True, include_gnn=False, include_site=True)
    names = [name for name, _, _ in plan]
    assert "fase0:download" not in names
    assert "fase1:backtest" in names
    assert "fase2:backtest" in names
    assert "fase3:conformal" in names
    assert "fase4:backtest" not in names
    assert "fase6:backtest" in names
    assert "fase5:export" in names

    rc = reproduce.main(["--skip-download", "--skip-gnn", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fase3:conformal" in out
    assert "fase5:export" in out
