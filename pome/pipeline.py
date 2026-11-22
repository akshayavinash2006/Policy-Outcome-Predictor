"""End-to-end prediction for one policy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import games as G
from .forecast import Forecast, base_rate, reference_class_forecast
from .index import VectorStore, load_or_build_index
from .llm import LLMForecast, PolicyLLM
from .retrieve import PolicyQuery, Precedent, candidate_pool, classify_game_knn, retrieve


@dataclass
class PredictionResult:
    query: PolicyQuery
    game: str
    game_source: str                       # "user", "llm" or "knn"
    game_distribution: dict[str, float]    # kNN vote, for reference
    game_rationale: str
    precedents: list[Precedent]
    forecast: Forecast                     # outside view (always available)
    llm_forecast: LLMForecast | None = None   # inside view (optional)
    invalid_citations: list[str] = field(default_factory=list)
    threshold: float = 0.0
    game_params: dict[str, float] | None = None
    param_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def margin(self) -> float | None:
        return G.margin(self.game, self.game_params)

    @property
    def binding_term(self) -> str | None:
        return G.binding_term(self.game, self.game_params)

    @property
    def final_probs(self) -> dict[str, float]:
        if self.llm_forecast and not self.forecast.abstained:
            return self.llm_forecast.probs()
        return self.forecast.probs

    @property
    def final_label(self) -> str:
        p = self.final_probs
        return max(p, key=p.get)


class PolicyPredictor:
    def __init__(self, store: VectorStore | None = None, use_llm: bool | None = None):
        self.store = store or load_or_build_index()
        if use_llm is None:
            use_llm = PolicyLLM.available()
        self.llm = PolicyLLM() if use_llm else None

    def predict(
        self,
        query: PolicyQuery,
        *,
        qvec: np.ndarray | None = None,
        pool: np.ndarray | None = None,
        hard_filters: dict[str, str] | None = None,
        k: int = C.TOP_K,
        threshold: float | None = None,
        use_llm: bool = True,
        llm_rerank: bool = False,
    ) -> PredictionResult:
        df = self.store.df
        if qvec is None:
            qvec = self.store.embed_query(query.text())
        if pool is None:
            pool = candidate_pool(df, hard_filters=hard_filters)
        outcome_pool = pool & df["has_outcome"].to_numpy()
        llm = self.llm if use_llm else None

        # 1. game model: user > LLM scorer > kNN vote of nearest precedents
        knn_game, knn_dist, knn_params = classify_game_knn(self.store, qvec, pool)
        reasons: dict[str, str] = {}
        if query.game:
            src, rationale = "user", "Specified by the analyst."
            if not G.clean_params(query.game, query.game_params):
                query.game_params = (knn_params if query.game == knn_game else
                                     classify_game_knn(self.store, qvec, pool, game=query.game)[2])
        else:
            choice = llm.score_game(query) if llm else None
            if choice:
                query.game, query.game_params = choice.game, choice.param_dict()
                src, rationale, reasons = "llm", choice.rationale, choice.reason_dict()
            else:
                query.game, query.game_params = knn_game, knn_params
                src = "knn"
                rationale = ("Similarity-weighted vote of the nearest precedents; parameters "
                             "are their weighted average.")
        query.game_params = G.clean_params(query.game, query.game_params)

        # 2. retrieval (game and margin now available for the soft boosts)
        precs = retrieve(self.store, query, qvec, outcome_pool, k=k, threshold=threshold)

        # 3. optional LLM re-rank: drop look-alikes judged a poor fit
        if llm and llm_rerank and precs:
            rr = llm.rerank(query, precs)
            if rr:
                fit = {i.record_id: i for i in rr.items}
                for p in precs:
                    if p.record_id in fit:
                        p.differs.append(f"LLM: {fit[p.record_id].differs}")
                        if fit[p.record_id].fit < 0.3:
                            p.above_threshold = False
                precs.sort(key=lambda p: -(fit[p.record_id].fit if p.record_id in fit else 0))

        # 4. outside view
        base = base_rate(df, outcome_pool)
        fc = reference_class_forecast(precs, base)

        # 5. inside view
        llm_fc, invalid = None, []
        if llm and not fc.abstained:
            usable = [p for p in precs if p.above_threshold]
            llm_fc = llm.predict(query, usable, G.describe(query.game, query.game_params, rationale),
                                 base, fc.probs)
            if llm_fc:
                valid = {p.record_id for p in usable}
                cited = [rid for c in llm_fc.expected_effects for rid in c.record_ids]
                cited += [rid for r in llm_fc.risks for rid in r.record_ids]
                invalid = sorted({rid for rid in cited if rid not in valid})

        return PredictionResult(
            query=query, game=query.game, game_source=src, game_distribution=knn_dist,
            game_rationale=rationale, precedents=precs, forecast=fc,
            llm_forecast=llm_fc, invalid_citations=invalid,
            threshold=self.store.threshold if threshold is None else threshold,
            game_params=query.game_params, param_reasons=reasons,
        )
