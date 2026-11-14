# Fase 6 — Plano de Pesquisa: Dívida Imunológica e Reemergência do DENV-3

## 1. Contexto Biológico e Epidemiológico

O vírus da dengue possui quatro sorotipos distintos (DENV-1, DENV-2, DENV-3 e DENV-4). A infecção
por um sorotipo confere imunidade permanente e esterilizante contra o mesmo sorotipo, mas apenas
proteção heteróloga cruzada temporária (estimada entre 3 a 12 meses) contra os demais. Após esse
período de proteção cruzada, a suscetibilidade a uma infecção secundária heteróloga é restaurada e,
em muitos casos, agravada por facilitação mediada por anticorpos (ADE — *Antibody-Dependent
Enhancement*).

No Brasil:
1. O sorotipo DENV-3 causou graves epidemias no início dos anos 2000 (especialmente entre 2002 e
   2007).
2. Entre 2009 e 2022, o cenário nacional foi amplamente dominado por DENV-1 e DENV-2, com breves
   incursões de DENV-4 a partir de 2011. Durante quase 15 anos consecutivos, o DENV-3 esteve
   virtualmente indetectável na maior parte das cidades brasileiras.
3. Como resultado, todas as coortes nascidas no Brasil a partir de 2008 cresceram sem nunca terem
   sido expostas ao DENV-3 (imunidade populacional zero para faixas etárias abaixo de 15 anos).
4. Em 2023 e 2024, a vigilância genômica da Fiocruz e do Ministério da Saúde detectou a
   reintrodução e rápida expansão de linhagens de DENV-3 em múltiplos estados (como SP, RJ, PR, MG).
   Essa reemergência coincidiu com a temporada recorde histórica de 2023-24.

## 2. A Hipótese da Dívida Imunológica

**Hipótese central**: Municípios com capacidade histórica de transmissão que atravessaram
prolongados períodos de calmaria epidemiológica acumularam grandes contingentes de indivíduos
suscetíveis ao DENV-3. A reintrodução viral nesses municípios dispara trajetórias epidêmicas mais
explosivas e menos contidas pela imunidade de rebanho prévia.

### Desafio de Dados
As bases públicas operacionais semanais (InfoDengue/SINAN) não dispõem de genotipagem ou sorotipagem
semanal contínua para os 5.570 municípios do Brasil. A maioria dos testes confirmatórios em nível
local são testes rápidos de antígeno NS1 ou testes sorológicos IgM, que não diferenciam o sorotipo.

### Solução Metodológica: Proxies Causais de Série Histórica
Modelar a dinâmica de acúmulo de suscetibilidade e dívida imunológica por meio de operadores
estritamente causais (defasagens e médias móveis retroativas) sobre as séries de incidência e
casos ao longo de janelas de 1 a 5 anos:

1. **Dívida Imunológica Quinquenal (`immune_debt`)**:
   $$\mathrm{debt}_i(t) = \overline{\mathrm{inc}}_{i, 260}(t) - \overline{\mathrm{inc}}_{i, 104}(t)$$
   Onde $\overline{\mathrm{inc}}_{i, W}(t)$ é a incidência média móvel das últimas $W$ semanas do
   município $i$. Valores altos refletem municípios que tiveram forte circulação há 3–5 anos, mas
   foram silenciosos nos últimos 2 anos (acúmulo de suscetíveis).

2. **Razão de Calmaria (`quiet_ratio`)**:
   $$\mathrm{quiet}_i(t) = \frac{1}{156} \sum_{k=0}^{155} \mathbb{I}[y_i(t-k) < 1.0]$$
   Fração de semanas com menos de 1 caso registrado nos últimos 3 anos. Mede o tempo de ausência
   de desafio imunológico continuado na população.

3. **Impulso de Reemergência (`reemergence_boost`)**:
   $$\mathrm{boost}_i(t) = \log(1 + y_i(t)) - \frac{1}{52} \sum_{k=0}^{51} \log(1 + y_i(t-k))$$
   Detecta o momento em que um município dormente experimenta um salto abrupto de casos em relação
   à sua média anual.

4. **Brecha de Suscetibilidade (`susceptible_gap`)**:
   $$\mathrm{gap}_i(t) = \max\left(0, \overline{\mathrm{inc}}_{i, 260}(t) - \overline{\mathrm{inc}}_{i, 52}(t)\right)$$
   Mede o déficit acumulado de incidência no ano corrente contra o potencial endêmico quinquenal.

## 3. Protocolo de Ablação e Validação

1. **Variantes avaliadas**:
   - `lgbm_own`: Série própria + estáticas.
   - `lgbm_season`: + Sazonalidade.
   - `lgbm_nb`: + Vizinhos e UF.
   - `lgbm`: + Clima (modelo de produção da Fase 2).
   - `lgbm_immunity`: + Grupo `immunity` (4 novas features causais).

2. **Critérios de Avaliação**:
   - WIS e WIS relativo ao baseline `naive_level` em $h \in \{1, 2, 4, 8\}$.
   - WIS logarítmico (sensibilidade a municípios pequenos e início de surto).
   - PR-AUC do alerta de surto (capacidade de antecipar transição epidêmica).
   - Cobertura empírica dos intervalos de predição de 50% e 90%.
   - Comportamento isolado na temporada do choque 2023-24 vs temporadas endêmicas típicas.

3. **Diretriz de Transparência Científica**:
   Conforme definido na especificação original (`SPEC.md`), resultados neutros ou negativos devem
   ser documentados com o mesmo rigor de um resultado positivo, evitando publicação seletiva.
