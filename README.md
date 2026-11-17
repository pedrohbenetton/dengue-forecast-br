# dengue-forecast-br

**Previsão probabilística semanal de casos de dengue para os ~5.570 municípios brasileiros**, com intervalos calibrados (Conformal Prediction) e avaliação temporal honesta.

Especificação detalhada em [SPEC.md](SPEC.md).

[![CI](https://github.com/pedrohbenetton/dengue-forecast-br/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/dengue-forecast-br/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](.python-version)
[![Web: MapLibre](https://img.shields.io/badge/Web-MapLibre%20%2B%20TopoJSON-teal.svg)](web/)

---

## Resumo Executivo

O desafio de prever dengue no Brasil geralmente foca em agregados estaduais. Este projeto desenvolve e avalia uma solução em **escala municipal completa** (5.570 municípios) cobrindo horizontes de 1 a 8 semanas à frente:

1. **Desempenho Preditivo**: Um modelo único global **LightGBM Quantílico** supera o melhor baseline (*persistência estratificada por patamar de casos*) em **todos** os horizontes ($h \in \{1, 2, 4, 8\}$) e em todas as 27 Unidades da Federação.
2. **Redução de WIS**: Redução consistente de 14% a 25% no *Weighted Interval Score* (WIS) em relação ao baseline de referência.
3. **Alerta Precoce de Surtos**: O PR-AUC para prever incidência explosiva (> 300 casos / 100 mil habitantes) em 8 semanas à frente salta de **0,23 para 0,43**.
4. **Calibração Conformal (CQR e ACI)**: Pós-processamento via *Conformalized Quantile Regression* com correção de amostra finita e *Adaptive Conformal Inference* (Gibbs & Candès, 2021), corrigindo a supercobertura em períodos normais e recuperando a cobertura nominal durante o surto recorde histórico de 2024.
5. **Aplicação Web Interativa**: Painel interativo com **MapLibre GL JS**, malha TopoJSON simplificada do Brasil, *timeline slider* e *fan charts* dinâmicos por município (hospedado no GitHub Pages).

---

## Resultados Comparativos

Avaliação temporal por origem móvel (*rolling origin*) em 4 temporadas de teste (2022–23, 2023–24, 2024–25 e 2025–26) em mais de 4,6 milhões de previsões individuais:

| Modelo | WIS h=1 | WIS h=2 | WIS h=4 | WIS h=8 | Cob. 90% | PR-AUC Alerta (h=8) |
|---|---|---|---|---|---|---|
| Ingênuo (persistência simples) | 2,42 | 4,02 | 6,55 | 10,81 | 91,2% | 0,201 |
| Ingênuo por nível de casos (referência) | 2,10 | 3,17 | 5,15 | 8,61 | 92,5% | 0,234 |
| LightGBM Global Quantílico | 1,80 | 2,53 | 3,84 | 6,43 | 92,0% | 0,434 |
| **LightGBM + CQR Estático** | **1,78** | **2,50** | **3,80** | **6,38** | **90,4%** | **0,436** |
| **LightGBM + ACI Adaptativo** | **1,77** | **2,48** | **3,79** | **6,35** | **90,1%** | **0,441** |

![Série Histórica e Análise Exploratória](reports/figures/eda_ano_recorde.png)

---

## Fases do Projeto

- [x] **Fase 0 — Ingestão e Validação**: Clientes assíncronos resilientes para APIs do InfoDengue e IBGE com cache local, painel consolidado em matriz contígua (`PanelData`) e [EDA completa](reports/eda.md).
- [x] **Fase 1 — Baselines & Harness de Avaliação**: Baselines probabilísticos (persistência simples, ingênuo por nível, sazonal e histórico por SE) e harness anti-vazamento em janelas expansivas ([relatório da fase 1](reports/fase1.md)).
- [x] **Fase 2 — LightGBM Global Quantílico**: 42 features causais (lags, médias móveis, sazonalidade, temperatura/umidade defasadas, vizinhança espacial IBGE) e ablação completa ([relatório da fase 2](reports/fase2.md)).
- [x] **Fase 3 — Calibração Conformal**: Implementação de CQR estático e adaptativo (ACI) com garantia de monotonicidade e preservação de mediana ([relatório da fase 3](reports/fase3.md)).
- [x] **Fase 4 — Rede em Grafo Espaço-Temporal (GNN)**: resultado negativo para o grafo. A convolução entre municípios vizinhos não melhora a previsão; uma rede multi-horizonte sem grafo reduz o WIS em 9% em h=8, concentrado no surto de 2023-24. O LightGBM segue como modelo servido ([relatório da fase 4](reports/fase4.md)).
- [x] **Fase 5 — Aplicação Web & Produção**: Dashboard estático interativo em `web/` com mapa coroplético MapLibre, *fan charts* municipais e pipeline de publicação contínua no GitHub Pages.
- [x] **Fase 6 — Pesquisa & Ablação de Dívida Imunológica (DENV-3)**: Modelagem causal de acúmulo de suscetibilidade e coortes não-imunes na reemergência de sorotipos dormentes ([plano](docs/fase6-plano.md) e [relatório completo](reports/fase6.md)).

---

## Camada Visual Interativa (Web)

O repositório inclui uma aplicação web moderna sem backend em [`web/`](web/):
* **Mapa Coroplético**: Renderiza os 5.570 municípios a 60 FPS com MapLibre GL JS e malha TopoJSON compacta (< 2,5 MB).
* **Controle Temporal**: Slider semanal com animação play/pause acompanhando a evolução espacial do surto.
* **Drill-down com Fan Charts**: Gráficos SVG interativos com leques de incerteza (50%, 80%, 90% e 95%) para cada município.
* **Benchmark Interativo**: Tabelas e métricas comparativas em tempo real.

---

## Instalação e Reprodução

O projeto utiliza [`uv`](https://docs.astral.sh/uv/) para gerenciamento de dependências e ambientes Python:

```bash
# 1. Clonar repositório e sincronizar ambiente
git clone https://github.com/pedrohbenetton/dengue-forecast-br.git
cd dengue-forecast-br
uv sync

# 2. Rodar a suíte completa de testes (115 testes, 100% offline)
uv run pytest -q
uv run ruff check

# 3. Pipeline de dados (APIs públicas)
uv run python -m dengue_forecast.download     # Download resiliente InfoDengue (~12 min)
uv run python -m dengue_forecast.panel        # Consolidação do painel (panel.parquet)
uv run python -m dengue_forecast.eda          # Figuras de análise exploratória

# 4. Backtest e Modelagem
uv run python -m dengue_forecast.backtest     # Avaliação dos baselines
uv run python -m dengue_forecast.backtest --models lgbm  # LightGBM quantílico
uv run python -m dengue_forecast.conformal    # Calibração conformal (CQR / ACI)
uv run python -m dengue_forecast.viz.phase3   # Diagnósticos de calibração

# 5. Frontend Web
uv run python -m dengue_forecast.export       # Exporta dados compactos para o site
cd web && npm install && npm run build        # Build da aplicação estática
```

---

## Fontes de Dados

* **[InfoDengue](https://info.dengue.mat.br)**: API pública `alertcity` (casos semanais, nowcasting, clima e incidência).
* **[IBGE Localidades e Malhas](https://servicodados.ibge.gov.br)**: Divisões político-administrativas e malha municipal TopoJSON.

Toda a pasta `data/` é mantida fora do controle de versão (`.gitignore`) e pode ser regenerada a partir dos scripts da biblioteca.

---

## Licença

Distribuído sob a licença [MIT](LICENSE).
