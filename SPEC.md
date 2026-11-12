# SPEC — Previsão probabilística de dengue por município (Brasil)

## Problema

Prever, toda semana, quantos casos de dengue cada um dos ~5.570 municípios brasileiros vai
notificar nas próximas semanas, com intervalos de incerteza calibrados, para apoiar priorização
de recursos (agentes de endemias, leitos, insumos).

O desafio Infodengue-Mosqlimate (IMDC) já cobre previsões por estado. Este projeto mira o nível
municipal, com foco em calibração e em avaliação temporal honesta.

## Alvo

- Unidade: (geocode IBGE, semana epidemiológica).
- `y` = casos notificados na semana (`casos` do InfoDengue).
- Horizontes `h ∈ {1, 2, 4, 8}` semanas à frente da semana de origem `t`.
- Saída: quantis `{0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975}`.

## Dados (fase 0)

| Fonte | Conteúdo | Acesso |
|---|---|---|
| InfoDengue `api/alertcity` | casos semanais, nowcast, Rt, população, clima (temp/umidade) por município | público, sem chave |
| IBGE localidades | lista de municípios, UF, região | público |
| IBGE malhas / geobr | adjacência entre municípios (features espaciais) | público |
| Mosqlimate datastore | clima ERA5 em lote | exige chave (opcional) |

Dados brutos ficam em `data/raw/` (não versionados); `data/processed/` guarda o painel
consolidado em parquet.

## Protocolo de avaliação (o coração do projeto)

- **Temporada** = SE 41 do ano Y-1 até SE 40 do ano Y.
- **Backtest por origem móvel**: temporadas de teste 2022-23, 2023-24 (recorde histórico),
  2024-25 e 2025-26. Para cada temporada, o modelo é treinado com tudo o que veio antes (janela
  expansiva). A temporada anterior à de teste serve de validação para hiperparâmetros.
- **Regra anti-vazamento**: toda feature de uma origem `t` usa apenas dados com semana ≤ `t`.
  Isso inclui clima: nada de clima "futuro" observado.
- **Métricas**:
  - WIS (Weighted Interval Score) e WIS relativo ao baseline (skill), que é a métrica principal;
  - MAE da mediana;
  - cobertura empírica dos intervalos de 50% e 90% (calibração);
  - PR-AUC para o evento "incidência > 300/100 mil na semana t+h" (alerta de surto).
- Resultados **estratificados** por faixa de população e por região. Média nacional esconde
  municípios pequenos cheios de zeros.

## Limitação conhecida

A API devolve a versão mais recente dos dados. As últimas semanas sofrem atraso de notificação,
e o backtest usa dados já consolidados, o que é otimista. Isso fica documentado, e as previsões
ao vivo usam `casos_est` (nowcast) nas semanas recentes.

## Fases

0. **Ingestão + validação dos dados**: cliente InfoDengue/IBGE com cache, painel consolidado, EDA.
1. **Baselines + harness de avaliação**: ingênuo (última semana), sazonal (mesma SE do ano
   anterior), média histórica por SE. O harness é testado antes de qualquer modelo.
2. **LightGBM global quantílico**: lags, médias móveis, sazonalidade, clima defasado, população e
   casos dos municípios vizinhos.
3. **Calibração conformal** dos intervalos (CQR por horizonte).
4. **Modelo espaço-temporal (GNN)**: só entra se superar a fase 2 no backtest.
5. **Produção + camada visual** (ver seção abaixo): pipeline semanal (GitHub Actions) que
   reconstrói os dados e publica o site.
6. **(Pesquisa) Dívida imunológica ao DENV-3**: a suscetibilidade por sorotipo melhora a
   previsão? Ablação no backtest de 2024–26; resultado negativo também é reportado.
7. **Write-up**: README em formato de paper curto, com números e com o que não funcionou.

## Camada visual (requisito de primeira classe, não enfeite)

Site estático no GitHub Pages: sempre no ar, sem cold start. Os dados (JSON/parquet) são gerados
pelo pipeline semanal.

- **Mapa coroplético animado** dos 5.570 municípios, com slider de semana e alternância entre
  observado, previsto (mediana) e probabilidade de surto. Geometrias em TopoJSON simplificado,
  renderizadas com MapLibre/deck.gl.
- **Drill-down por município**: fan chart com os quantis previstos para cada horizonte sobre a
  série observada, e as previsões passadas do backtest ("o que o modelo dizia na época").
- **Painel de avaliação**: heatmap de skill (WIS relativo) por UF × horizonte, diagrama de
  calibração (cobertura nominal vs. empírica), histograma PIT, curvas PR do alerta e ranking dos
  modelos.
- **Análise de erro**: onde e quando o modelo erra (início de surto, municípios pequenos, 2024),
  com mapa de resíduos.
- **Figuras estáticas** de qualidade de publicação (matplotlib) para o README e o write-up, com
  paleta e tipografia consistentes entre o site e as figuras.

## Hardware

O desenvolvimento acontece num PC com RTX 2060 (6 GB). As fases 0–3 rodam bem em CPU; a GPU
entra na fase 4 (PyTorch Geometric) e, opcionalmente, no LightGBM/XGBoost.

## Critérios de pronto

- `uv run pytest` verde, e ruff limpo no CI.
- Um comando reproduz o backtest completo a partir do zero.
- A tabela final compara todos os modelos contra os baselines, por horizonte e por estrato.
