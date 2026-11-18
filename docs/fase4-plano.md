# Fase 4 — Plano de Arquitetura: Modelo Espaço-Temporal em Grafos (GNN)

## 1. Motivação e Critério de Decisão

Na [Fase 2](../reports/fase2.md), o LightGBM quantílico global alcançou expressiva redução de WIS
(14% a 25% abaixo do melhor baseline) usando árvores de decisão tabulares. No entanto, a ablação
revelou que a inclusão ingênua de features espaciais (incidência média de vizinhos contíguos agregada
em uma única feature escalar `nb0`) praticamente não melhorou o WIS bruto em $h=8$ (0,727 vs 0,740).

Árvores tabulares tratam o espaço de forma desacoplada: cada nó de decisão divide apenas uma feature
univariada por vez. Redes Neurais em Grafos Espaço-Temporais (ST-GNNs — *Spatio-Temporal Graph Neural Networks*),
por outro lado, operam difusão convolucional contínua ao longo da malha de adjacência geográfica,
permitindo capturar frentes de onda de propagação viral entre municípios conurbados e corredores rodoviários.

> **Regra do Projeto (`SPEC.md`)**:
> O modelo espaço-temporal **só entra para a fase de produção se superar o LightGBM da Fase 2 no backtest**.
> Caso contrário, o resultado negativo é formalmente documentado e o LightGBM permanece como modelo final.

---

## 2. Topologia do Grafo e Estrutura dos Dados

O grafo municipal brasileiro é definido como $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$:
* **Vértices $\mathcal{V}$**: $N \approx 5.570$ municípios (geocodes IBGE).
* **Arestas $\mathcal{E}$**: Contiguidade territorial (*Queen adjacency*) derivada da malha simplificada IBGE.
* **Matriz de Adjacência $\mathbf{A} \in \{0, 1\}^{N \times N}$**: Simétrica, com auto-laços $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$.
* **Laplaciano Normalizado**:
  $$\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-1/2}, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$

### Tensores de Entrada
Para uma semana de origem $t$ e janela histórica de retrocesso $T = 12$ semanas:
* $\mathbf{X}_t \in \mathbb{R}^{N \times T \times F}$: Estado municipal ao longo das últimas $T$ semanas.
  * $F \approx 6$ canais dinâmicos: $\log(1 + y)$, taxa de incidência, anomalia de temperatura, anomalia de umidade, proporção de zeros, impulso de reemergência.
* $\mathbf{S} \in \mathbb{R}^{N \times F_s}$: Features estáticas municipais (log população, faixa de porte, UF em embedding).

---

## 3. Arquitetura da Rede (ST-GNN)

A arquitetura desacopla a propagação espacial da evolução temporal em blocos espaço-temporais:

```
[Entrada: N x T x F]
        │
        ▼
┌──────────────────────────────────────────────┐
│  Bloco Temporal 1: Conv1D Causal / GRU       │ (Filtro temporal ao longo de T)
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Bloco Espacial: Graph Convolution (GCN/GAT) │ (Mensagens entre vizinhos A_norm)
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Bloco Temporal 2: GRU Recorrente            │ (Resumo da dinâmica recente)
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Cabeça Decodificadora Multi-Quantílica     │ (Linear -> N x H x 9 quantis)
└──────────────────────────────────────────────┘
```

### Decodificação Quantílica e Função de Perda (Pinball Loss)
O modelo projeta diretamente a variação $\Delta_{t, h} = \log(1 + y_{t+h}) - \log(1 + y_t)$ para os 9 quantis nominais
$\tau \in \{0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975\}$ através da perda multi-quantílica Pinball:

$$\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}}) = \sum_{h \in \mathcal{H}} \sum_{q \in \mathcal{Q}} \frac{1}{N} \sum_{i=1}^N \max\left(q (y_{i, t+h} - \hat{y}_{i, t+h}^q), \; (q - 1)(y_{i, t+h} - \hat{y}_{i, t+h}^q)\right)$$

Ordenação pós-rede garante monotonicidade: $\hat{y}_{i, h}^{q_1} \le \hat{y}_{i, h}^{q_2}$ para $q_1 < q_2$.

---

## 4. Orçamento de Memória e Treinamento na RTX 2060 (6 GB VRAM)

* **Grafo Completo**: $N = 5.570$ nós.
* **Memória de um passo temporal**:
  * $5.570 \text{ nós} \times 12 \text{ semanas} \times 64 \text{ canais ocultos} \times 4 \text{ bytes (FP32)} \approx 17,1 \text{ MB}$.
  * Com gradientes e buffers do otimizador AdamW, uma passagem direta e reversa para o grafo nacional completo consome **menos de 350 MB de VRAM**.
* **Capacidade da GPU**: A RTX 2060 (6 GB GDDR6) tem capacidade de folga superior a 15× para acomodar o grafo nacional simultaneamente em batch único por passo temporal, eliminando a necessidade de amostragem estocástica de subgrafos (*Cluster-GCN* ou *GraphSAINT*).
* **Treinamento**:
  * Otimizador: `AdamW(lr=1e-3, weight_decay=1e-4)`.
  * Scheduler: `CosineAnnealingLR` com warmup de 5 épocas.
  * Épocas: 50 a 100 com Early Stopping na temporada de validação 2021-22.
  * Tempo estimado de treino na RTX 2060: ~15 a 25 minutos por fold.

---

## 5. Critérios de Sucesso no Backtest

| Métrica | Benchmark a Bater (LightGBM Fase 2) | Meta da GNN |
|---|---|---|
| **WIS h=1** | 1,80 | $\le 1,75$ |
| **WIS h=8** | 6,43 | $\le 6,20$ |
| **PR-AUC Surto h=8** | 0,434 | $\ge 0,460$ |
| **Início de Surto 2024** | Subestimação do pico | Acompanhamento da subida via difusão |
