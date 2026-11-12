# dengue-forecast-br

Previsão probabilística semanal de casos de dengue para os ~5.570 municípios brasileiros, com
intervalos calibrados e avaliação temporal honesta. Especificação completa em [SPEC.md](SPEC.md).

**Status:** fases 0 a 2 concluídas — ingestão, painel e [EDA](reports/eda.md); harness de
avaliação e [baselines](reports/fase1.md); [LightGBM global quantílico](reports/fase2.md). Este
README vira um paper curto ao final (fase 7).

Resultado até aqui (4 temporadas de teste, 5.570 municípios): o LightGBM reduz o WIS em 14% a
25% contra o melhor baseline em todos os horizontes e vence em todas as 27 UFs. O PR-AUC do
alerta de surto em 8 semanas vai de 0,23 para 0,43. Vizinhos e clima não ajudaram; a subida de
um surto recorde ainda é subestimada.

| Modelo | WIS h=1 | WIS h=2 | WIS h=4 | WIS h=8 |
|---|---|---|---|---|
| Ingênuo por nível (melhor baseline) | 2,10 | 3,17 | 5,15 | 8,61 |
| LightGBM | 1,80 | 2,53 | 3,84 | 6,43 |

![2024 foi o maior ano de dengue da série](reports/figures/eda_ano_recorde.png)

## Uso

```bash
uv sync
uv run pytest -q
uv run ruff check

uv run python -m dengue_forecast.download   # ~12 min, retomável; --refresh-years 2 para atualizar
uv run python -m dengue_forecast.panel      # data/processed/panel.parquet
uv run python -m dengue_forecast.eda        # reports/figures/*.png
uv run python -m dengue_forecast.backtest   # backtest dos baselines (~1 min)
uv run python -m dengue_forecast.backtest --models lgbm   # LightGBM (~1 h em CPU)
```

## Dados

- [InfoDengue](https://info.dengue.mat.br) `api/alertcity` — casos, nowcast, Rt, clima.
- [IBGE localidades](https://servicodados.ibge.gov.br/api/docs/localidades) — municípios.

`data/` não é versionado; tudo é reproduzível a partir das APIs públicas.
