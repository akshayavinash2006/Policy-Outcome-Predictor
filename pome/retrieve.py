"""Precedent retrieval: filter -> cosine search -> soft metadata boost -> gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import games as G
from .data import design_text
from .index import VectorStore


@dataclass
class PolicyQuery:
    """A new (or held-out) policy described only by its design."""
    policy_name: str
    description: str
    stated_goal: str = ""
    target_group: str = ""
    sector: str = ""
    sub_sector: str = ""
    mechanism_type: str = ""
    level: str = ""
    state: str = ""
    year_start: int | None = None
    game: str | None = None           # one of games.GAMES; filled by the scorer if None
    game_params: dict | None = None   # that game's parameters in [0, 1]

    def text(self) -> str:
        return design_text(asdict(self))

    @property
    def margin(self) -> float | None:
        return G.margin(self.game, self.game_params) if self.game else None

    @classmethod
    def from_record(cls, r: dict) -> "PolicyQuery":
        yr = r.get("year_start")
        return cls(
            policy_name=r["policy_name"], description=r["description"],
            stated_goal=r["stated_goal"], target_group=r["target_group"],
            sector=r["sector"], sub_sector=r["sub_sector"],
            mechanism_type=r["mechanism_type"], level=r["level"], state=r["state"],
            year_start=None if pd.isna(yr) else int(yr),
        )


@dataclass
class Precedent:
    record: dict
    similarity: float          # raw cosine (used for the relevance gate)
    score: float               # cosine + metadata boosts (used for ranking)
    above_threshold: bool
    shared: list[str] = field(default_factory=list)
    differs: list[str] = field(default_factory=list)
    weight: float = 0.0        # filled by the forecaster

    @property
    def record_id(self) -> str:
        return self.record["record_id"]


def candidate_pool(
    df: pd.DataFrame,
    *,
    before_year: int | None = None,
    exclude_family: str | None = None,
    hard_filters: dict[str, str] | None = None,
) -> np.ndarray:
    """Boolean mask of records allowed as precedents.

    before_year: strict date cutoff for backtesting; undated rows are dropped
    because we can't prove they precede the target.
    exclude_family: drop every row describing the same underlying policy."""
    mask = np.ones(len(df), dtype=bool)
    if before_year is not None:
        yrs = df["year_start"]
        mask &= (yrs.notna() & (yrs < before_year)).to_numpy(dtype=bool)
    if exclude_family is not None:
        mask &= (df["family"] != exclude_family).to_numpy()
    for col, val in (hard_filters or {}).items():
        if val:
            mask &= (df[col] == val).to_numpy()
    return mask


def _compare(q: PolicyQuery, r: dict) -> tuple[list[str], list[str]]:
    shared, differs = [], []
    for f, label in [("sector", "sector"), ("mechanism_type", "mechanism"),
                     ("game", "game"), ("level", "level of government")]:
        qv = getattr(q, f, None)
        if not qv:
            continue
        (shared if qv == r.get(f) else differs).append(
            f"{label}: {r.get(f)}" if qv == r.get(f) else f"{label}: {qv} vs {r.get(f)}"
        )
    if q.state and r.get("state") and q.state != r["state"]:
        differs.append(f"state: {q.state} vs {r['state']}")
    return shared, differs


def retrieve(
    store: VectorStore,
    query: PolicyQuery,
    qvec: np.ndarray,
    pool: np.ndarray,
    k: int = C.TOP_K,
    threshold: float | None = None,
    boosts: dict[str, float] | None = None,
) -> list[Precedent]:
    threshold = store.threshold if threshold is None else threshold
    boosts = C.BOOSTS if boosts is None else boosts
    df = store.df
    sims = store.similarities(qvec)
    score = sims.copy()
    for col, b in boosts.items():
        qv = getattr(query, col, None)
        if qv:
            score = score + b * (df[col].to_numpy() == qv)
    qm = query.margin
    if qm is not None and C.MARGIN_BOOST:
        rm = df["margin"].to_numpy(dtype=float)
        close = np.where(np.isnan(rm), 0.0, 1 - np.abs(qm - rm) / 2)
        score = score + C.MARGIN_BOOST * close
    score = np.where(pool, score, -np.inf)
    order = np.argsort(-score)[:k]
    out = []
    for i in order:
        if not np.isfinite(score[i]):
            break
        rec = df.iloc[i].to_dict()
        shared, differs = _compare(query, rec)
        out.append(Precedent(rec, float(sims[i]), float(score[i]),
                             bool(sims[i] >= threshold), shared, differs))
    return out


def classify_game_knn(
    store: VectorStore, qvec: np.ndarray, pool: np.ndarray, k: int = 7,
    game: str | None = None,
) -> tuple[str, dict[str, float], dict[str, float] | None]:
    """Similarity-weighted vote of the nearest neighbours' game.

    The no-LLM fallback: returns the winning game, the vote distribution, and
    the similarity-weighted mean parameters of the neighbours in that game
    (None if none of them has been scored). With game given, it returns the
    weighted mean parameters of the nearest scored precedents of that game."""
    df = store.df
    sims = np.where(pool, store.similarities(qvec), -np.inf)
    idx = [i for i in np.argsort(-sims)[:k] if np.isfinite(sims[i])]
    votes: dict[str, float] = {}
    wts = {i: float(np.exp(sims[i] / C.SIM_TEMPERATURE * 0.5)) for i in idx}
    for i in idx:
        g = df.iloc[i]["game"]
        votes[g] = votes.get(g, 0.0) + wts[i]
    total = sum(votes.values()) or 1.0
    dist = {g: v / total for g, v in sorted(votes.items(), key=lambda x: -x[1])}
    if game is None:
        game = next(iter(dist)) if dist else "principal_agent"
        members = [i for i in idx if df.iloc[i]["game"] == game and df.iloc[i]["game_scored"]]
    else:   # forced game: average the nearest scored precedents of that game
        cand = [i for i in np.argsort(-sims) if np.isfinite(sims[i])
                and df.iloc[i]["game"] == game and df.iloc[i]["game_scored"]][:k]
        wts.update({i: float(np.exp(sims[i] / C.SIM_TEMPERATURE * 0.5)) for i in cand})
        members = cand
    params = None
    if members:
        w = np.array([wts[i] for i in members])
        params = {s: float(np.dot(w, [df.iloc[i]["game_params"][s] for i in members]) / w.sum())
                  for s in G.GAMES[game].symbols}
    return game, dist, params
