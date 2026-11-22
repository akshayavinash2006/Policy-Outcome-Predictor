# Method

## 1. The forecasting problem

A decision maker considering a new policy asks: *what happened the last time
something like this was tried?* We treat this as a **prediction policy
problem** (Kleinberg et al., 2015): the decision needs a good forecast of the
outcome, not a causal estimate of a single parameter.

The forecast target is the outcome class of a policy: `improved`, `mixed`,
`no_clear_effect` or `worsened`. We also report a **net-outcome index**,
E[score] with improved = +1, mixed / no clear effect = 0, worsened = -1.

## 2. Outside view first: reference-class forecasting

Planners systematically over-weight the specifics of their own project and
under-weight how similar projects actually turned out. This is the planning
fallacy (Kahneman & Lovallo, 1993). The remedy is **reference-class
forecasting** (Flyvbjerg, 2006): pick a class of comparable past cases, read
off their outcome distribution, and only then adjust for case specifics.

The retrieval-augmented system implements this directly:

| Step | Reference-class forecasting | Implementation |
|---|---|---|
| 1 | Identify the reference class | Embed the policy's *design* text; cosine search over 57 precedents; soft boosts for the same game, a similar incentive margin, instrument and sector |
| 2 | Establish the outcome distribution | Similarity- and evidence-weighted outcome shares, shrunk toward the corpus base rate |
| 3 | Adjust for the specific case | Optional LLM step (Groq): starts from the step-2 distribution and must justify every move away from it, citing precedents |

### Weighting

Each precedent *i* gets weight

    w_i = exp((s_i - s_max) / T)  x  evidence_i

* `s_i` is the cosine similarity of design texts and `T` a temperature
  (0.05), so the closest analogues dominate.
* `evidence_i` follows an **evidence hierarchy** in the spirit of the
  Maryland Scientific Methods Scale (Sherman et al., 1998): RCT 1.00,
  quasi-experimental 0.85, CAG audit 0.70, official data 0.60, study 0.55,
  news 0.35. This is multiplied by the coder's quality rating
  (high 1.0 / medium 0.75 / low 0.5) and the verification status.

### Bayesian shrinkage

With so few precedents, raw shares are noisy. We use a Dirichlet-multinomial
posterior:

    P(c) = (a * base_c + sum_i w_i * 1[y_i = c]) / (a + sum_i w_i)

where `base_c` is the Laplace-smoothed outcome share in the visible corpus
and `a = 2` is the prior strength in pseudo-observations. When the evidence
is thin, the forecast falls back toward the base rate instead of
over-reacting to one precedent.

### Confidence and abstention

* **Relevance gate:** a precedent counts only if its similarity exceeds
  the 80th percentile of all pairwise similarities in the corpus. With fewer
  than two such precedents, the system returns *insufficient evidence*
  instead of forcing an analogy.
* **Effective sample size:** n_eff = (sum w)^2 / sum w^2 (Kish), the
  number of equally weighted precedents the evidence is worth.

## 3. Game model: three games, one inequality each

A policy works only if the people it depends on find it in their interest to
act as intended. We model that with **three classic games**. Every policy is
assigned to exactly one, the interaction that will most decide its outcome,
and each game reduces to a single inequality, the **incentive margin** m:

| Game | Players | Policy succeeds iff | Absorbs the older categories |
|---|---|---|---|
| **G1 Principal-agent** | government -> the agent whose hidden action matters (official, intermediary, firm, taxpayer, beneficiary) | m = w + p·F − g ≥ 0 | principal-agent, tax compliance, conditional incentives, mechanism design |
| **G2 Collective action** | many users of a shared resource, network or standard | m = t − g + k·x₀ ≥ 0 | commons / externality, coordination |
| **G3 Commitment** | government -> public, market or organised group asked to pay a cost now | m = κ·b − c ≥ 0 | credibility / signalling, bargaining |

### G1 Principal-agent (incentive compatibility)

| Agent's payoff | Detected (p) | Not detected (1 − p) |
|---|---|---|
| Comply | w | w |
| Deviate | g − F | g |

Deviating pays g − p·F in expectation, so the agent complies iff
w ≥ g − p·F. Parameters: w reward for complying, p probability deviation is
detected, F penalty if caught, g gain from deviating (rents, evasion,
shirking). Critical detection rate: p* = (g − w)/F. (Jensen & Meckling,
1976; Becker, 1968; Allingham & Sandmo, 1972.)

### G2 Collective action (prisoner's dilemma and stag hunt)

With a share x of others cooperating, a user's gain from cooperating over
defecting is t − g + k·x, where t is the policy's correction (tax, quota,
metering, subsidy for cooperating), g the private gain from defecting, and k
the extra payoff when others cooperate. Evaluated at the expected starting
share x₀:

* t − g + k < 0: defection dominates, a **prisoner's dilemma** (tragedy of
  the commons);
* t − g < 0 < t − g + k: a **stag hunt** with tipping point x* = (g − t)/k;
  the policy works only if x₀ starts above it;
* t − g ≥ 0: cooperation dominates; the policy has closed the gap.

(Hardin, 1968; Olson, 1965; Ostrom, 1990; Schelling, 1960.)

### G3 Commitment (time consistency)

| Public's payoff | Government keeps the policy (κ) | Government reverses (1 − κ) |
|---|---|---|
| Respond | b − c | −c |
| Wait | 0 | 0 |

The public responds iff κ·b ≥ c: credibility κ must exceed κ* = c/b.
Parameters: κ credibility (legislated vs announced, track record, power of
groups that could force a climb-down), b benefit of responding, c cost of
responding now. (Kydland & Prescott, 1977; Barro & Gordon, 1983;
Rubinstein, 1982.)

### Scoring

Each parameter is scored on a 0-1 scale (0 none, 0.25 weak, 0.5 moderate,
0.75 strong, 1 overwhelming); values are relative, not money. m is clipped to
[−1, 1]. The **binding constraint** is the parameter whose move to its
favourable extreme would raise m the most, holding the others fixed (for G1:
Δw = 1 − w, Δp = (1 − p)F, ΔF = p(1 − F), Δg = g). It names the design lever
with the largest payoff.

**Protocol.** `scripts/score_games.py` asks the LLM to choose the game and
score its parameters for every record, with a one-line reason per score,
into `game_params.csv`. The researchers then review every row, correct it,
and mark it `reviewed = y`; re-running the script never overwrites reviewed
rows. For a new policy the same prompt scores it at forecast time; offline,
the game is a similarity-weighted vote of the nearest precedents and the
parameters are their weighted average.

**Leakage guard.** Scores are made from the *design* fields only (name,
sector, instrument, level, target group, description, goal). The scorer
never sees the outcome, so the margin can't encode the answer and can be
used in the backtest.

### How the game model is used

1. **Explanation**: the memo shows the payoff table, the parameter scores
   with reasons, the margin with the numbers substituted, the regime (e.g.
   "stag hunt, below the tipping point") and the levers ranked by margin gain.
   The app draws each inequality as a payoff diagram: comply vs deviate
   against detection p (crossing at p*), the gain from cooperating against
   the share of others cooperating (crossing at the tipping point x*, with an
   arrow for whether cooperation spreads or unravels), and respond vs wait
   against credibility (crossing at kappa*). A bar chart shows the margin
   gain from fixing each parameter, and the analytics tab plots every
   record's margin against its real outcome (`pome/game_plots.py`).
2. **Retrieval**: precedents in the same game get a +0.06 ranking boost, and
   precedents with a similar margin get up to +0.04 (ranking only; the
   relevance gate still uses raw similarity).
3. **Testing**: the backtest includes a *game model* baseline that predicts
   the outcome from the margin and game alone, and the analytics report the
   rank correlation between margin and net outcome. If the model has no
   predictive skill, the results say so.

The evidence-weighted reference-class forecast itself does not use the
margin, so a weak game model cannot degrade it.

## 4. Management output: the decision memo

The output is a one-page memo: bottom line with probabilities against the
base rate; the game model with its parameters, margin and binding constraint; the reference
class table with similarities, weights, evidence grades and sources; what
separated the successful precedents from the failures; a **risk register**
(risk, likelihood, mitigation, evidence); and design levers. It is decision
support. It is not an automated decision.

## 5. Evaluation design

**Backtest.** Each past policy is hidden and predicted from the rest.

Leakage controls:
1. *Family exclusion*: all rows about the same underlying policy (e.g. two
   evaluations of Delhi odd-even) are removed together.
2. *Date cutoff* (strict mode): only policies that started strictly earlier
   are visible; undated rows are dropped.
3. *Design-only queries*: outcome text never enters the query embedding.

**Baselines.**
* *Climatology / majority*: always predict the visible base rate. This is
  the bar any forecaster must clear.
* *Random precedents*: same forecaster, precedents drawn at random. Tests
  whether *retrieval* adds value.
* *Game base rate*: outcome shares of same-game policies. Tests whether
  the game category alone predicts.
* *Game model*: multinomial logistic regression of the outcome on the
  incentive margin and game, refit on each visible pool. Tests whether the
  mathematical model alone predicts.
* *No evidence weights* (ablation): tests the evidence hierarchy.
* *Closed-book LLM*: the same LLM without retrieval.

**Metrics.** Accuracy and macro-F1 (4-class); direction accuracy (up /
flat-or-mixed / down); multi-class **Brier score**, a proper scoring rule
(Brier, 1950), and the **Brier skill score** against climatology; log loss;
coverage; retrieval hit rate; game accuracy (kNN and LLM vs the reviewed game); and, with the LLM, citation
validity and LLM-judged citation faithfulness. 95% bootstrap intervals
throughout, because n is small.

**Sensitivity analysis.** The hyperparameters (prior strength, temperature,
k) are fixed a priori in `config.py`. A grid over them is *reported*, not
used to pick values, so the headline numbers involve no selection on the
test set.

## 6. Limitations

* 57 records (47 distinct policies). Wide confidence intervals are expected
  and reported honestly.
* Outcome labels compress heterogeneous metrics into four classes;
  quantitative range forecasts aren't evaluated, because `outcome_value`
  isn't on a common scale.
* Precedents are correlational evidence about *other* contexts.
* Game assignments and parameter scores are modelling judgements (LLM-drafted,
  human-reviewed), not claims made by the sources; the 0-1 scale is ordinal.

## References

* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1).
* Flyvbjerg, B. (2006). From Nobel Prize to project management: getting risks right. *Project Management Journal*, 37(3).
* Kahneman, D., & Lovallo, D. (1993). Timid choices and bold forecasts. *Management Science*, 39(1).
* Kleinberg, J., Ludwig, J., Mullainathan, S., & Obermeyer, Z. (2015). Prediction policy problems. *American Economic Review*, 105(5).
* Kish, L. (1965). *Survey Sampling*. Wiley.
* Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
* Sherman, L. W. et al. (1998). *Preventing Crime: What Works, What Doesn't, What's Promising*. NIJ Research in Brief.
* Aamodt, A., & Plaza, E. (1994). Case-based reasoning. *AI Communications*, 7(1).
