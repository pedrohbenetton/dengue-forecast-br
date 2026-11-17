# Fase 4 — Rede em grafo espaço-temporal (GNN)

Reproduzir (a partir do painel da fase 0 e das previsões das fases 1 e 2; cerca de 2 minutos por
temporada e modelo numa RTX 2060):

```bash
uv sync --group gnn                           # PyTorch com CUDA 12.4 (opcional; o CI não instala)
uv run --group gnn python -m dengue_forecast.backtest --models gnn,gnn_nograph \
    --seasons 2022 --forecast-dir data/forecasts/val          # validação 2021-22
uv run --group gnn python -m dengue_forecast.backtest \
    --models naive,naive_level,lgbm,gnn,gnn_nograph --out-dir reports/fase4
```

Tabelas completas em [`fase4/`](fase4/) (WIS relativo calculado contra `naive`).

## Resumo

A pergunta da fase era se a troca de mensagens entre municípios vizinhos melhora a previsão.
**Não melhora**: a mesma rede sem o grafo é igual ou melhor no teste. O que ajuda é a rede em si:
uma rede neural multi-horizonte sobre as mesmas features do LightGBM reduz o WIS em 8% a 9% em
h=8 e empata em h=1. O ganho se concentra no surto de 2023-24 e nos municípios grandes; em
2024-25 e 2025-26 os três modelos empatam.

## Modelo

- **Entrada**: as mesmas 42 features causais da fase 2 (`FeatureBuilder`), com as seis features
  sazonais repetidas para cada horizonte; a UF entra por embedding. Padronização calculada só no
  treino; ausentes viram zero depois de padronizar.
- **Rede** (`STGNN`): projeção linear, duas camadas residuais que concatenam o estado do nó com a
  média normalizada dos vizinhos (`Â = D^-½ (A + I) D^-½`), e uma cabeça que devolve 4 horizontes
  × 9 quantis de uma vez. 128 unidades, dropout 0,1.
- **Alvo e perda**: variação em escala log, como no LightGBM; perda pinball, com máscara onde o
  alvo ainda não foi observado. Quantis ordenados na saída.
- **Treino**: grafo nacional inteiro por passo, 8 origens por lote, 30 épocas, AdamW (1e-3) com
  decaimento em cosseno. Um ajuste por temporada, com o painel truncado na véspera.
- **Controle** (`gnn_nograph`): a mesma rede com a mensagem dos vizinhos zerada. Ela ainda vê as
  features de vizinhança da fase 2 (`nb0`, `uf0` etc.); o que se remove é só a convolução.

## Protocolo

Toda a escolha foi feita na dobra de validação (treino até 2020-21, avaliação em 2021-22). A
configuração acima foi a primeira testada e não foi ajustada; as temporadas de teste rodaram uma
única vez, com uma semente. Não há parada antecipada: o número de épocas é fixo.

Validação, WIS em h=8, três sementes ([`validacao_sementes.csv`](fase4/validacao_sementes.csv);
a semente 0 é a do backtest):

| Modelo | Semente 0 | Semente 1 | Semente 2 | Média |
|---|---|---|---|---|
| gnn | 3,197 | 3,254 | 3,244 | 3,232 |
| gnn_nograph | 3,276 | 3,317 | 3,280 | 3,291 |
| lgbm | 3,499 | | | |

Na validação o grafo ganhava cerca de 2% em h=8, do mesmo tamanho da variação entre sementes
(0,03 a 0,06). O teste não confirmou essa vantagem.

## Resultado no teste (2022-23 a 2025-26)

| h | WIS lgbm | WIS gnn | WIS gnn_nograph | PR-AUC lgbm | PR-AUC gnn | PR-AUC gnn_nograph |
|---|---|---|---|---|---|---|
| 1 | 1,799 | 1,823 | 1,786 | 0,901 | 0,903 | 0,903 |
| 2 | 2,527 | 2,492 | 2,474 | 0,840 | 0,842 | 0,841 |
| 4 | 3,844 | 3,749 | 3,674 | 0,719 | 0,720 | 0,718 |
| 8 | 6,433 | 5,925 | 5,846 | 0,434 | 0,467 | 0,459 |

WIS em h=8 por temporada:

| Temporada | lgbm | gnn | gnn_nograph |
|---|---|---|---|
| 2022-23 | 3,88 | 3,44 | 3,54 |
| 2023-24 | 15,84 | 14,15 | 13,54 |
| 2024-25 | 4,06 | 4,11 | 4,25 |
| 2025-26 | 2,05 | 2,07 | 2,12 |

- **O grafo não paga**. Em 2023-24, a temporada que motivou a fase, a rede sem grafo é 4% melhor
  que a com grafo em h=8. Nas outras três a rede com grafo é 2% a 3% melhor. No agregado, a sem
  grafo vence em todos os horizontes por 1% a 2%; a dispersão entre sementes não foi medida no
  teste, então essa diferença não distingue os dois.
- **A rede ganha do LightGBM onde há surto**: 9% a 15% em h=8 em 2022-23 e 2023-24, e 14% a 17%
  nos municípios acima de 500 mil habitantes. Nas duas temporadas calmas o LightGBM é igual ou
  ligeiramente melhor.
- **Calibração**: a cobertura de 90% fica em 91,5% a 93,7% no agregado para os três, e cai para
  cerca de 87% em 2023-24 h=8 também nos três. O intervalo de 50% da rede cobre 56% a 62%
  (LightGBM: 63% a 69%).

## Metas do plano

| Meta ([plano](../docs/fase4-plano.md)) | gnn | Atingida |
|---|---|---|
| WIS h=1 ≤ 1,75 | 1,823 | não |
| WIS h=8 ≤ 6,20 | 5,925 | sim |
| PR-AUC h=8 ≥ 0,460 | 0,467 | sim |
| Acompanhar a subida de 2024 pela difusão | sem grafo é melhor em 2023-24 | não |

## Veredito

Pela regra do SPEC, o modelo espaço-temporal só entra se superar o LightGBM. A hipótese
espacial foi rejeitada: a convolução no grafo não acrescenta nada além do que as features de
vizinhança já davam (resultado coerente com a ablação da fase 2). O LightGBM continua sendo o
modelo servido pelo pipeline e pela camada web.

A rede sem grafo é uma candidata a substituí-lo em horizontes longos, mas essa troca não foi
feita aqui: a vantagem vem de duas das quatro temporadas, foi medida com uma semente, e a
calibração conformal da fase 3 teria de ser refeita sobre ela.

## Limitações

- Uma semente no teste; a dispersão só foi medida na validação.
- Nenhuma busca de hiperparâmetros para a rede, e uma única arquitetura de grafo (contiguidade,
  duas camadas). Grafos de mobilidade ou atenção não foram testados.
- O treino em CUDA não é determinístico bit a bit; os testes de determinismo rodam em CPU.
