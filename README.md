# Policy Outcome Predictor

Retrieval-augmented **reference-class forecasting** of Indian policy outcomes.
For a proposed policy it finds the most similar documented precedents, weights
them by how credible their evidence is, models the key players' incentives with one of three games, and
produces a cited **decision memo with a risk register**.

Major project · Aditya Sridhar (1RV24CS018), Akshay Avinash (1RV24CS027) · Mentor: Dr. Bindu Ashwini

---

## Why this is an economics and management project, not just a RAG demo

| Idea | Where it comes from | Where it lives in the code |
|---|---|---|
| **Outside view first.** Forecast from the outcome distribution of comparable past cases, then adjust | Kahneman & Lovallo (1993), *Management Science*; Flyvbjerg (2006), reference-class forecasting | `pome/forecast.py` |
| **Evidence hierarchy.** An RCT counts for more than a news report | Maryland Scientific Methods Scale (Sherman et al., 1998) | `config.EVIDENCE_TYPE_WEIGHT` |
| **Bayesian shrinkage to the base rate.** Thin evidence can't produce extreme forecasts | Dirichlet-multinomial posterior | `reference_class_forecast()` |
| **Prediction policy problems.** Many decisions need a good forecast, not a causal parameter | Kleinberg et al. (2015), *AER* | framing |
| **Game model.** Three games (principal-agent, collective action, commitment), each an inequality on 0-1 parameters giving an incentive margin and a binding constraint | Jensen-Meckling, Becker, Allingham-Sandmo; Hardin, Olson, Ostrom, Schelling; Kydland-Prescott | `pome/games.py`, `game_params.csv` |
| **Proper scoring rules.** Forecasts are judged by Brier score and skill against climatology, not just accuracy | Brier (1950) | `pome/evaluate.py` |
| **Decision memo + risk register.** The output is decision support for a manager | project / risk management practice | `pome/memo.py` |
| **Selective prediction.** Abstain ("insufficient evidence") when no precedent is close enough | relevance gate + coverage curve | `retrieve.py`, `evaluate.coverage_curve` |

The full write-up is in [docs/METHOD.md](docs/METHOD.md); the app shows it in its *Method* tab.

---

## Quick start (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts\build_index.py          # embed the 57 records
.\.venv\Scripts\python.exe scripts\score_games.py          # LLM-draft game_params.csv, then review it
.\.venv\Scripts\streamlit.exe run app.py                   # the app
.\.venv\Scripts\python.exe scripts\backtest.py             # evaluation -> artifacts/backtest/
.\.venv\Scripts\python.exe scripts\analyze.py              # statistics -> artifacts/analytics.md
.\.venv\Scripts\python.exe scripts\predict.py --no-llm --name "..." --description "..." --sector energy_water_environment --mechanism subsidy --out memo.md
```

**LLM layer (optional, Groq).** Copy `.env.example` to `.env` and set
`GROQ_API_KEY` (free at console.groq.com). This turns on LLM game-model scoring, optional re-ranking, the
"inside view" adjustment with cited claims, an LLM-written risk register,
and, in the backtest (`--llm`), the closed-book LLM baseline and
LLM-judged citation faithfulness. Without a key, everything runs offline.

---

## Pipeline (matches the six stages in the proposal)

| Stage | Implementation |
|---|---|
| 1-2 Data and records | `policy_records.csv` (25 fields) -> `pome/data.py` derives policy families, evidence weights, and design-only text |
| 3 Game model | `game_params.csv`: game + 0-1 parameters per record, LLM-drafted from design text (`scripts/score_games.py`) and human-reviewed; for new policies: analyst, LLM, or nearest-neighbour vote |
| 4 Indexing | `all-MiniLM-L6-v2` embeddings of design text (TF-IDF fallback); exact cosine search over NumPy (at N=57 this beats any ANN index; FAISS/Chroma is a drop-in at scale) |
| 5 Retrieval and prediction | filter -> cosine -> soft boosts (game, incentive margin, instrument, sector) -> relevance gate (80th percentile of pairwise similarity) -> evidence-weighted reference class -> optional LLM adjustment with citation validation |
| 6 Evaluation | leakage-controlled backtest against five baselines, bootstrap CIs, sensitivity grid |

---

## Results (offline, MiniLM embeddings, no LLM)

Backtest with **family exclusion** (all rows about the same policy are held out together) and design-only queries.

**Strict date cutoff:** 40 dated targets, each predicted only from policies that started earlier.

| Method | Accuracy (95% CI) | Coverage | Macro-F1 | Brier ↓ | Brier skill vs base rate |
|---|---|---|---|---|---|
| Reference class (system) | 0.30 (0.17 to 0.45) | 0.53 | **0.20** | 0.775 | -0.02 |
| No evidence weights (ablation) | 0.25 (0.12 to 0.40) | 0.53 | 0.14 | 0.783 | -0.03 |
| Game base rate (game category only) | 0.23 (0.10 to 0.38) | 1.00 | 0.12 | 0.797 | -0.05 |
| Game model (margin + game, logistic) | 0.30 (0.17 to 0.45) | 1.00 | 0.19 | **0.745** | **+0.02** |
| Random precedents | 0.23 (0.10 to 0.38) | 1.00 | 0.12 | 0.773 | -0.02 |
| Base rate (climatology) | 0.33 (0.17 to 0.47) | 1.00 | 0.13 | 0.759 | 0 |

**Leave-one-family-out:** 55 targets.

| Method | Accuracy (95% CI) | Coverage | Macro-F1 | Brier ↓ | Brier skill |
|---|---|---|---|---|---|
| Reference class (system) | 0.44 (0.31 to 0.56) | 0.91 | 0.28 | 0.715 | +0.00 |
| No evidence weights | 0.42 (0.29 to 0.55) | 0.91 | 0.26 | 0.723 | -0.01 |
| Game base rate | 0.45 (0.33 to 0.58) | 1.00 | 0.23 | 0.734 | -0.02 |
| Game model (margin + game, logistic) | 0.49 (0.36 to 0.62) | 1.00 | **0.34** | **0.697** | **+0.03** |
| Random precedents | 0.42 (0.29 to 0.55) | 1.00 | 0.15 | 0.711 | +0.01 |
| Base rate | 0.42 (0.29 to 0.55) | 1.00 | 0.15 | 0.717 | 0 |

*Game-model numbers use LLM-drafted parameters that have not yet been human-reviewed; re-run after review.*

### How to read this (an honest summary)

1. **With 57 records, nothing statistically beats the base rate on outcome
   accuracy.** All confidence intervals overlap. The 27-setting sensitivity
   grid (`artifacts/backtest/*_sensitivity.csv`) shows this is not a tuning
   artefact: Brier skill ranges from -0.15 to +0.01. This matches the
   forecasting literature: base rates are hard to beat, which is the point of
   the outside view.
2. **The evidence hierarchy helps.** Weighting by evidence quality lowers the
   Brier score against the unweighted ablation in both modes.
3. **The system discriminates where the base rate can't.** Its macro-F1 is
   higher (0.20-0.26 vs 0.13-0.15), because the majority rule always says
   "mixed".
4. **Abstention is part of the design.** Under the strict cutoff, the
   system declines 47% of early policies for lack of earlier analogues. That
   is a correct "insufficient evidence", not a failure.
5. **The game model has a small but consistent signal.** Predicting from the
   incentive margin and game alone gives the best Brier score in both modes
   (skill +0.02 to +0.03), though the intervals still overlap the base rate.
   The no-LLM game classifier (nearest-neighbour vote) matches the LLM's game
   38-58% of the time against 3 classes. Run `backtest.py --llm` with an API
   key for the LLM game accuracy and the LLM-RAG vs closed-book comparison.

The value of the system is therefore **transparent, evidence-graded
precedent retrieval and structured risk analysis**, not an oracle. Say so in
the viva; it is a strength.

### Findings from the evidence base (`artifacts/analytics.md`)

* **Who evaluates shapes the verdict.** CAG audits rated 0 of 10 policies
  "improved" (95% CI 0-28%), against 50-67% for quasi-experimental studies
  and RCTs. This is a direct argument for the evidence hierarchy.
* **Instruments:** subsidies had the best net outcome (+0.55, 6/11
  improved, 0 worsened); bans (-0.50) and regulation (-0.25) the worst.
* **Game model:** the incentive margin, scored from design only, is
  positively rank-correlated with the net outcome (Spearman rho = +0.28,
  permutation p = 0.034, n = 55). Policies with m >= 0 average a net outcome
  of +0.28 (n = 32), those with m < 0 average -0.17 (n = 23). This is the one
  association below p < 0.05; it rests on unreviewed LLM scores, so confirm it
  after the human review.
* No categorical factor (game, instrument, sector, evidence type, level)
  reaches p < 0.05 under permutation tests, so treat those as hypotheses.

---

## Project layout

```
policy_records.csv      the dataset (57 rows x 25 fields)
pome/
  config.py             every modelling assumption, in one place
  data.py               loading, policy families, evidence weights, design text
  games.py              three games: payoffs, margin, binding constraint, levers
  game_plots.py         payoff diagrams, lever chart, margin-vs-outcome scatter (Plotly)
  index.py              embeddings + vector store
  retrieve.py           candidate pool, search, boosts, gate, kNN game vote
  forecast.py           reference-class forecaster (Bayesian, evidence-weighted)
  llm.py                Groq LLM: game scoring, re-rank, cited forecast, faithfulness judge
  pipeline.py           end-to-end prediction
  memo.py               decision memo / risk register
  evaluate.py           backtest, baselines, metrics, coverage curve, sensitivity
  analytics.py          permutation tests, Wilson intervals, outcome breakdowns
scripts/                build_index, predict, backtest, analyze
app.py                  Streamlit UI (forecast, backtest, analytics, data, method)
docs/METHOD.md          methodology write-up with references
artifacts/              generated results (backtest CSV/JSON, analytics, example memo)
```
