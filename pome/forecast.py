"""Reference-class forecasting.

The "outside view" (Kahneman & Lovallo, 1993; Flyvbjerg, 2006) predicts a
new case from the distribution of outcomes in a class of similar past cases,
not from the case's own plausible-sounding story.

Here:
  1. the reference class is the set of retrieved precedents above the
     relevance gate;
  2. each precedent i gets weight
         w_i = sim_rel_i * evidence_i
     where sim_rel_i = exp((s_i - s_max) / T) favours the closest analogues
     and evidence_i comes from the evidence hierarchy (RCT > ... > news);
  3. the weighted outcome shares are shrunk toward the corpus base rate:
         P(c) = (a * base_c + sum_i w_i * 1[y_i = c]) / (a + sum_i w_i)
     a Dirichlet-multinomial posterior with prior strength a.

The Kish effective sample size n_eff = (sum w)^2 / sum w^2 is how many
equally weighted precedents the evidence is worth, and drives confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .retrieve import Precedent


@dataclass
class Forecast:
    probs: dict[str, float]
    predicted: str
    expected_score: float          # net-outcome index in [-1, 1]
    n_precedents: int              # above-threshold precedents with known outcomes
    n_eff: float
    top_similarity: float
    confidence: str                # high / medium / low / insufficient
    abstained: bool
    base_rate: dict[str, float] = field(default_factory=dict)
    note: str = ""


def base_rate(df: pd.DataFrame, pool: np.ndarray | None = None) -> dict[str, float]:
    """Laplace-smoothed outcome shares in the candidate pool (the 'climatology')."""
    sub = df if pool is None else df[pool]
    counts = sub["outcome_direction"].value_counts()
    raw = np.array([counts.get(c, 0) + 1.0 for c in C.OUTCOME_CLASSES])
    return dict(zip(C.OUTCOME_CLASSES, raw / raw.sum()))


def _expected(p: dict[str, float]) -> float:
    return float(sum(p[c] * C.OUTCOME_SCORE[c] for c in C.OUTCOME_CLASSES))


def from_distribution(probs: dict[str, float], base: dict[str, float], note: str = "",
                      abstained: bool = False) -> Forecast:
    pred = max(probs, key=probs.get)
    return Forecast(probs, pred, _expected(probs), 0, 0.0, 0.0,
                    "insufficient" if abstained else "low", abstained, base, note)


def reference_class_forecast(
    precedents: list[Precedent],
    base: dict[str, float],
    prior_strength: float = C.PRIOR_STRENGTH,
    temperature: float = C.SIM_TEMPERATURE,
    min_precedents: int = C.MIN_PRECEDENTS,
    use_evidence_weights: bool = True,
    gate: bool = True,
) -> Forecast:
    usable = [p for p in precedents
              if p.record.get("has_outcome") and (p.above_threshold or not gate)]
    top_sim = max((p.similarity for p in precedents), default=0.0)

    if len(usable) < min_precedents:
        fc = from_distribution(
            dict(base), base, abstained=True,
            note=(f"Only {len(usable)} precedent(s) cleared the relevance gate; "
                  "returning the base rate. Treat as insufficient evidence."),
        )
        fc.top_similarity = top_sim
        fc.n_precedents = len(usable)
        return fc

    s = np.array([p.similarity for p in usable])
    sim_rel = np.exp((s - s.max()) / temperature)
    ev = np.array([p.record["evidence_weight"] if use_evidence_weights else 1.0 for p in usable])
    w = sim_rel * ev
    for p, wi in zip(usable, w):
        p.weight = float(wi)

    mass = {c: 0.0 for c in C.OUTCOME_CLASSES}
    for p, wi in zip(usable, w):
        mass[p.record["outcome_direction"]] += wi
    denom = prior_strength + w.sum()
    probs = {c: (prior_strength * base[c] + mass[c]) / denom for c in C.OUTCOME_CLASSES}

    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    gap = top_sim - (min(p.similarity for p in usable))
    if n_eff >= 3 and w.sum() >= prior_strength:
        conf = "high" if probs[max(probs, key=probs.get)] >= 0.5 else "medium"
    elif n_eff >= 2:
        conf = "medium"
    else:
        conf = "low"

    pred = max(probs, key=probs.get)
    return Forecast(
        probs=probs, predicted=pred, expected_score=_expected(probs),
        n_precedents=len(usable), n_eff=n_eff, top_similarity=top_sim,
        confidence=conf, abstained=False, base_rate=base,
        note=f"{len(usable)} precedents, effective sample size {n_eff:.1f}, similarity spread {gap:.2f}.",
    )
