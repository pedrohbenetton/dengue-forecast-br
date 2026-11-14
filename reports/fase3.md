# Fase 3 — Calibração Conformal dos Intervalos de Previsão (CQR e ACI)

Reproduzir:

```bash
uv run python -m dengue_forecast.conformal \
    --forecast-dir data/forecasts \
    --val-dir data/forecasts/val \
    --out-dir reports/fase3 \
    --reference naive_level
uv run python -m dengue_forecast.viz.phase3
```

## Resumo

A calibração conformal (Conformalized Quantile Regression — CQR) atua como camada de pós-processamento
sobre as previsões quantílicas do LightGBM, corrigindo as distorções empíricas identificadas na
[fase 2](fase2.md) sem necessidade de retreino do modelo:

1. **Intervalos largos demais em temporadas normais**: a cobertura de 90% caía entre 92% e 94% (conservadora em excesso);
2. **Subestimação em quebras de regime**: no ano recorde histórico de 2023-24, a cobertura de 90% em $h=8$ caía para 87,2%.

A calibração conformal ajusta as margens simetricamente em escala log1p com preservação estrita da mediana,
garantindo não-negatividade e monotonicidade dos quantis. Com a variante adaptativa (ACI — *Adaptive Conformal Inference*, Gibbs & Candès 2021),
o modelo monitora os erros semana a semana e alarga os intervalos dinamicamente durante surtos fora de escala.

## Formulação Matemática

Para cada nível de intervalo central nominal $1 - \alpha \in \{0,50, 0,80, 0,90, 0,95\}$, definimos o escore de conformidade CQR em escala log1p:

$$s_i = \max\left(\log(1 + \hat{q}_{\alpha/2}) - \log(1 + y_i), \; \log(1 + y_i) - \log(1 + \hat{q}_{1 - \alpha/2})\right)$$

O escore é estritamente negativo quando a observação $y_i$ cai dentro do intervalo previsto, e positivo quando há extrapolação.

### Correção de Amostra Finita
A margem conformal $m$ é calculada no quantil de amostra finita sobre os $n$ erros observados:

$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil$$

Se $k > n$, utiliza-se $\max(s)$; se $n=0$, $m = 0$.

### Aplicação das Margens e Monotonicidade
Para cada nível nominal com índices $(lo, hi)$:

$$\hat{q}_{lo}' = \operatorname{expm1}\left(\log(1 + \hat{q}_{lo}) - m\right)$$
$$\hat{q}_{hi}' = \operatorname{expm1}\left(\log(1 + \hat{q}_{hi}) + m\right)$$

A mediana ($\hat{q}_{0.5}$) permanece inalterada. Em seguida, aplica-se clipagem $\ge 0$ e imposição de monotonicidade não decrescente de dentro para fora:

$$\hat{q}_i = \min(\hat{q}_i, \hat{q}_{i+1}) \quad \text{para } i \in \{3, 2, 1, 0\}$$
$$\hat{q}_i = \max(\hat{q}_i, \hat{q}_{i-1}) \quad \text{para } i \in \{5, 6, 7, 8\}$$

## Variantes Avaliadas

| Variante | Tipo | Estrato | Comportamento Temporal |
|---|---|---|---|
| `lgbm_cqr` | CQR Estático | Horizonte ($h$) | Margem fixa calculada na temporada anterior |
| `lgbm_cqr_level` | CQR Estático | Horizonte $\times$ Nível de casos recente | Margem estratificada por atividade epidêmica recente |
| `lgbm_aci` | CQR Adaptativo | Horizonte ($h$) | Atualização online do $\alpha_{\mathrm{eff}}$ semana a semana |
| `lgbm_aci_level` | CQR Adaptativo | Horizonte $\times$ Nível | Adaptação online local por patamar de casos |

### Regra Anti-Vazamento da ACI
Na semana de origem $t$, o estado $\alpha_{\mathrm{eff}}$ é atualizado apenas com as previsões emitidas em origens anteriores cujo alvo já foi observado ($\text{target} = t \le t$).
Nenhuma informação futura de $y$ na semana $t+h$ é utilizada para emitir o ajuste da origem $t$.

## Visualizações

O módulo `dengue_forecast.viz.phase3` gera as seguintes figuras de diagnóstico:
- `reports/figures/fase3_calibracao.png`: Diagrama de calibração empírica vs nominal.
- `reports/figures/fase3_cobertura_semanal_2024.png`: Evolução da cobertura de 90% semana a semana em 2024.
- `reports/figures/fase3_alpha_eff_aci.png`: Trajetória do $\alpha_{\mathrm{eff}}$ evidenciando a reação ao surto.
- `reports/figures/fase3_skill_wis.png`: WIS relativo ao baseline ingênuo por nível.
