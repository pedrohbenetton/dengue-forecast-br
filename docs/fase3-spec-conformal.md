Escreva DOIS arquivos Python COMPLETOS (código final, sem reticências, sem "..."), para o repositório cujo código você pode ler (leia antes: src/dengue_forecast/backtest.py, metrics.py, baselines.py, data.py, epiweek.py, tests/conftest.py, tests/test_backtest.py). Estilo: Python 3.12, ruff line-length 100, docstrings e comentários em português, numpy/pandas vetorizados, sem dependências novas.

Responda SOMENTE com os dois arquivos, cada um num bloco ```python precedido de uma linha `### path`. Nada mais.

=== Arquivo 1: src/dengue_forecast/conformal.py ===

Contexto: as previsões do backtest são DataFrames com colunas
geocode, season, origin (YYYYWW), target (YYYYWW), horizon (1,2,4,8), band, region, population, y, crosses_year,
e os quantis em backtest.QCOLS = ["q0.025","q0.05","q0.1","q0.25","q0.5","q0.75","q0.9","q0.95","q0.975"] (contagens de casos, >= 0).

Conteúdo exigido:

1. `INTERVALS: dict[float, tuple[int, int]] = {0.5: (3, 5), 0.8: (2, 6), 0.9: (1, 7), 0.95: (0, 8)}` — nível nominal -> índices (inferior, superior) em QCOLS. A mediana (índice 4) nunca é alterada.

2. `conformity_scores(df) -> np.ndarray` [n, 4], na ordem de INTERVALS: escore CQR em escala log1p,
   s = max(log1p(lo) - log1p(y), log1p(y) - log1p(hi)). Negativo quando y está dentro do intervalo.

3. `conformal_quantile(scores: np.ndarray, nominal: float) -> float`: quantil com correção de amostra finita,
   nível k = ceil((n + 1) * nominal); se n == 0 devolve 0.0; se k > n devolve o máximo dos escores; senão o k-ésimo menor (np.partition).

4. `apply_margins(q: np.ndarray, margins: np.ndarray) -> np.ndarray`: q [n, 9] em contagens, margins [n, 4] em log1p.
   Para cada nível i com índices (lo, hi): lo' = expm1(log1p(q_lo) - m), hi' = expm1(log1p(q_hi) + m). Mediana inalterada.
   Depois: clip em >= 0 e impor monotonicidade de dentro para fora: para os índices 3,2,1,0 (inferiores) cada um = min(ele, o vizinho mais interno);
   para 5,6,7,8 (superiores) cada um = max(ele, o vizinho mais interno). Isso garante q não decrescente e preserva a mediana. Não modifica a entrada.

5. `level_bin` vem de `dengue_forecast.baselines` (np.digitize em LEVEL_EDGES). Função
   `attach_origin_cases(df, data: PanelData) -> pd.DataFrame`: devolve cópia com colunas `origin_cases` (casos observados na semana de origem, de data.cases) e `level` (= level_bin(origin_cases)). Use mapeamento geocode->linha e epiweek->coluna; levante ValueError se algum (geocode, origin) não existir.

6. `class StaticCQR`: `__init__(self, by: tuple[str, ...] = ("horizon",))`; `fit(self, cal: pd.DataFrame) -> "StaticCQR"` calcula, para cada grupo de `by` e cada nível, a margem `conformal_quantile(scores do grupo, nível)`; guarda também a margem global por horizonte como fallback para grupos ausentes no teste. `transform(self, df) -> pd.DataFrame`: cópia de df com QCOLS ajustados. Nome do modelo: atributo `name` passado no __init__ (default "lgbm_cqr").

7. `class AdaptiveCQR` (Adaptive Conformal Inference, Gibbs & Candès 2021, aplicado por grupo):
   `__init__(self, gamma: float = 0.05, by=("horizon",), name="lgbm_aci")`.
   `run(self, cal: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame`:
   - Os escores de calibração (cal) por grupo e nível ficam fixos (ordenados).
   - Para cada grupo e nível, estado alpha_eff começa em alpha = 1 - nominal.
   - Percorra as semanas-origem do teste em ordem crescente. Na origem t, primeiro atualize o estado com os erros das linhas do teste cujo `target` == t (alvos que acabaram de ser observados; podem ter sido emitidas em origens anteriores): err = fração de linhas do grupo com y fora do intervalo AJUSTADO que foi emitido para elas; alpha_eff += gamma * (alpha - err). Se não houver linhas com target == t no grupo, não atualize.
   - Depois emita o ajuste para todas as linhas com origin == t: margem = conformal_quantile dos escores de calibração do grupo no nível efetivo 1 - alpha_eff, com 1 - alpha_eff truncado em [0, 1] (nível 1 -> máximo; nível 0 -> mínimo dos escores).
   - Guarde os quantis ajustados de cada linha (são os usados para o erro quando o alvo for observado).
   - REGRA ANTI-VAZAMENTO: o ajuste emitido na origem t só pode depender de y de linhas com target <= t. Garanta isso pela ordem do laço.
   - Devolve cópia do test com QCOLS ajustados (mesma ordem de linhas da entrada) e uma coluna extra float `alpha_eff_90` com o alpha efetivo do nível 0.9 usado em cada linha.

8. `def run(...)` + `main(argv)` CLI:
   - argumentos: --forecast-dir (default data/forecasts), --val-dir (default data/forecasts/val), --panel (default data/processed/panel.parquet), --out-dir (default reports/fase3), --gamma (0.05), --reference (default "naive_level").
   - Para cada temporada S em backtest.TEST_SEASONS: teste = read_parquet(forecast_dir/f"lgbm_{S}.parquet"); calibração = previsões lgbm da temporada S-1 (para S=2023 vem de val_dir/"lgbm_2022.parquet"; senão forecast_dir/f"lgbm_{S-1}.parquet").
   - Anexe origin_cases/level aos dois com attach_origin_cases (carregue o painel com backtest.load_data).
   - Variantes: "lgbm_cqr" = StaticCQR(by=("horizon",)); "lgbm_cqr_level" = StaticCQR(by=("horizon","level")); "lgbm_aci" = AdaptiveCQR(gamma, by=("horizon",)); "lgbm_aci_level" = AdaptiveCQR(gamma, by=("horizon","level")).
   - Para cada variante: concatene as temporadas, `backtest.score(...)`, `.assign(model=nome)[backtest._KEEP]` com floats em float32 como em backtest.run_all, e salve em forecast_dir/"scored"/f"{nome}.parquet". Salve também os quantis ajustados de 2024 de cada variante em forecast_dir/f"{nome}_2024.parquet" (para fan charts) — inclua origin_cases, level e alpha_eff_90 quando houver.
   - Ao final: `backtest.write_tables(backtest.load_scored(forecast_dir, [reference, "lgbm", *variantes]), Path(out_dir), reference)` e imprima tables["por_horizonte"].round(3).
   - `if __name__ == "__main__": raise SystemExit(main())`. Docstring do módulo com "Uso: uv run python -m dengue_forecast.conformal".

=== Arquivo 2: tests/test_conformal.py ===

Testes pytest, sem rede, rápidos (< 2 s no total), usando dados sintéticos pequenos montados no próprio teste (helper que gera um DataFrame de previsões com as colunas acima; quantis gerados de forma monótona, ex.: mediana m e quantis m * exp(z_k * s)). Testes exigidos:
- test_conformity_scores_sign: y dentro -> escore negativo; y abaixo/acima -> positivo, valor exato em log1p.
- test_conformal_quantile_finite_sample: n=9, nominal 0.9 -> k=ceil(9)=9 -> maior escore; n=0 -> 0.0; n=4, nominal 0.5 -> k=3 -> terceiro menor.
- test_apply_margins_monotone_nonnegative_median_fixed: margens positivas, negativas grandes e mistas; resultado não decrescente por linha, >= 0, mediana igual à da entrada, entrada não modificada.
- test_static_cqr_hits_nominal_on_exchangeable_data: cal e teste da mesma distribuição (miscalibrada de propósito: intervalos estreitos demais); após o ajuste, a cobertura empírica do intervalo de 90% no teste fica em [0.87, 0.93] e a de 50% em [0.46, 0.54] (use n grande, ex. 20_000 linhas, seed fixa).
- test_static_cqr_shrinks_too_wide_intervals: intervalos largos demais -> margens negativas e cobertura cai para perto do nominal.
- test_static_cqr_fallback_for_unseen_group: grupo de `level` ausente na calibração usa a margem do horizonte.
- test_aci_no_leakage: rode AdaptiveCQR; depois altere y de todas as linhas com target > T (T no meio do teste) multiplicando por 50; rode de novo; os quantis ajustados das linhas com origin <= T devem ser idênticos; e algum das linhas com origin > T deve mudar.
- test_aci_adapts_to_regime_shift: na segunda metade das origens do teste, y passa a ser bem maior que a mediana prevista (choque); a cobertura de 90% da ACI na segunda metade deve ser maior que a do StaticCQR na mesma metade, e alpha_eff_90 deve cair (ficar abaixo de 0.1) ao fim.
- test_attach_origin_cases: usa a fixture toy_data de tests/conftest.py; confere origin_cases e level para 2 linhas e o ValueError para geocode inexistente.
