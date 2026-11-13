# Fase 3 — plano aprovado e estado em 2026-11-13

Plano aprovado pelo autor em 2026-11-12. **Nada da fase 3 foi implementado ainda**: não existe
`src/dengue_forecast/conformal.py` nem `tests/test_conformal.py`.

## Objetivo

Calibrar os intervalos do LightGBM com conformal (CQR), como pós-processamento das previsões já
salvas, sem retreinar o modelo. Problemas a corrigir (ver [fase2.md](../reports/fase2.md)):

- cobertura do intervalo de 90% em 92% a 94% nas temporadas normais (larga demais);
- cobertura de 90% em 87,2% em h=8 no ano recorde 2023-24 (estreita demais).

## De onde vêm os erros de calibração (sem vazamento)

- **Início da temporada**: erros fora da amostra da temporada anterior. Para calibrar 2022-23 é
  preciso rodar o LightGBM em 2021-22.
- **Durante a temporada**: na origem `t`, entram só os pares cujo alvo já foi observado
  (alvo ≤ `t`).

## Variantes a comparar

Todas ajustam em escala log, por horizonte e por nível de intervalo, com a mediana fixa. A
margem pode ser negativa (estreita o intervalo).

| Nome | Método |
|---|---|
| `lgbm_cqr` | CQR estático, margem da temporada anterior, por horizonte |
| `lgbm_cqr_level` | idem, por horizonte × nível de casos na origem (`baselines.level_bin`) |
| `lgbm_aci` | CQR adaptativo (ACI, Gibbs & Candès 2021), atualizado a cada semana observada |
| `lgbm_aci_level` | ACI por horizonte × nível |

Depois do ajuste, os quantis são corrigidos para não se cruzarem nem ficarem negativos.

## Decisões já tomadas

- **Zeros**: não forçar quantis a zero. A cobertura de 50% acima da nominal nos municípios
  pequenos é esperada quando P(y=0) > 50% (o intervalo central é [0, 0]). Reportar a cobertura
  também só nas previsões com intervalo não degenerado. Essa explicação ainda **não foi
  verificada** separando os zeros.
- **Estratificação**: pelo nível recente de casos, não pelo porte do município.

## Especificação do módulo

[fase3-spec-conformal.md](fase3-spec-conformal.md) descreve `conformal.py` função a função
(escores, quantil com correção de amostra finita, aplicação das margens, `StaticCQR`,
`AdaptiveCQR`, CLI) e os nove testes exigidos, incluindo o anti-vazamento da ACI.

## Entregáveis

`reports/fase3.md` com tabela de WIS e cobertura antes e depois (por horizonte, temporada e
faixa), diagrama de calibração e cobertura de 90% em 2023-24 semana a semana.

## Para retomar em outra máquina

`data/` não é versionado; tudo se reconstrói (tempos medidos num PC de 12 núcleos):

```bash
uv sync
uv run python -m dengue_forecast.download      # ~12 min
uv run python -m dengue_forecast.panel
uv run python -m dengue_forecast.backtest --models naive,naive_level,seasonal,climatology  # ~1 min
uv run python -m dengue_forecast.backtest --models lgbm                                    # ~1 h
# conjunto de calibração da temporada 2022-23:
uv run python -m dengue_forecast.backtest --models lgbm --seasons 2022 --forecast-dir data/forecasts/val
```

O download traz a versão atual dos dados do InfoDengue, então os números podem diferir um pouco
dos relatórios (as semanas recentes são revisadas).

## Nota sobre o Gemini

A delegação ao Gemini via Antigravity (`agy` em modo headless) parou em 2026-11-12 com erro 403
("service has been disabled in this account for violation of Terms of Service"). O uso
interativo do Antigravity seguia funcionando. Não depender dessa delegação até isso se resolver.
