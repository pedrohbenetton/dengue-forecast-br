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

Tabelas completas em [`fase3/`](fase3/). Cerca de 4 minutos em CPU.

> **Correção (17/11/2026).** A primeira versão deste relatório e do README trazia números de
> calibração (WIS 6,38 e 6,35 em h=8, cobertura de 90% em 90,1–90,4%) que não vieram de nenhuma
> execução: a fase 3 não tinha sido rodada sobre os dados reais. Os números abaixo são os da
> execução real, e as figuras foram regeradas.

## Resumo

A calibração conformal é um pós-processamento das previsões quantílicas do LightGBM, sem
retreino. A motivação vinha da [fase 2](fase2.md): intervalos de 90% que cobriam 92–94% no
agregado, mas só 87,2% em 2023-24 h=8.

**Resultado: ganho pequeno, e só numa das quatro variantes.**

| Variante | WIS h=1 | h=2 | h=4 | h=8 | Cobertura 90% (média dos h) | PR-AUC h=8 |
|---|---|---|---|---|---|---|
| LightGBM sem calibração | 1,799 | 2,527 | 3,844 | 6,433 | 93,1% | 0,434 |
| CQR por horizonte | 1,799 | 2,527 | 3,844 | 6,448 | 93,2% | 0,434 |
| **CQR por horizonte × nível** | 1,802 | 2,528 | **3,802** | **6,296** | 93,5% | **0,440** |
| ACI por horizonte | 1,801 | 2,528 | 3,842 | 7,167 | 93,3% | 0,436 |
| ACI por horizonte × nível | 1,795 | 2,504 | 3,950 | 6,517 | 93,5% | 0,440 |

- **A cobertura agregada não chega a 90% porque o problema não é uniforme.** Os municípios
  pequenos, cheios de zeros, ficam acima do nominal (94–95% abaixo de 5 mil habitantes), enquanto
  os de 100–500 mil ficam abaixo (87,8% em h=8). Uma margem única por horizonte não corrige as
  duas coisas ao mesmo tempo: o CQR por horizonte quase não muda nada.
- **Estratificar por nível de casos é o que funciona.** Em h=8, o CQR por nível leva a cobertura
  de 90% dos municípios de 100–500 mil de 87,8% para 90,8%, e a dos acima de 500 mil de 90,4%
  para 92,9%, reduzindo o WIS em 2% (6,43 → 6,30). É a variante servida no site.
- **Nenhuma variante evita a queda de cobertura na subida de 2024.** Todas caem para cerca de
  75% na pior semana; a calibração usa a temporada anterior, que não tinha um surto daquele
  tamanho. Em 2023-24 h=8 a cobertura de 90% vai de 87,2% para 88,3% com o CQR por nível.
- **O ACI piora o horizonte longo.** Em h=8, o retorno do erro chega 8 semanas atrasado: o
  $\alpha_{\mathrm{eff}}$ continua caindo depois do pico e os intervalos ficam largos demais na
  descida. O WIS de 2023-24 h=8 sobe de 15,8 para 18,8.
- **Por que a margem por horizonte quase não age.** Entre 48% e 64% das linhas de calibração
  (conforme a temporada) têm zero casos e quantil inferior zero, e o escore de conformidade
  empata em exatamente 0. Com isso o quantil 90% dos escores é 0 na margem global de todas as
  temporadas, e as margens por horizonte ficam em zero ou perto disso (só h=8 muda um pouco em
  algumas temporadas). Os grupos com casos na origem têm escores informativos, e por isso a
  estratificação por nível é a que age.

As margens servidas em produção estão em [`fase3/margens_cqr_nivel.csv`](fase3/margens_cqr_nivel.csv),
calculadas com a temporada 2025-26. Como foi uma temporada calma, elas *estreitam* os intervalos
dos municípios com muitos casos; num ano de surto, isso tende a subcobrir.

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
