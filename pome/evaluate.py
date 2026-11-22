"""Backtest: hide each past policy's outcome, predict it from other policies,
compare.

Leakage controls
  * Family exclusion: rows describing the same underlying policy (e.g. two
    evaluations of Delhi odd-even) are never precedents for each other.
  * Date cutoff (mode="cutoff"): only policies that started strictly before
    the target are visible. Undated rows are excluded from the pool and from
    the targets, since we can't prove they come first.
  * Queries are built from design fields only; outcome text never enters
    the query embedding. Game parameters are scored from the same design
    fields, so the target's own scores are a fair stand-in for scoring a new
    policy at launch.

Methods compared
  rcf             the system: evidence-weighted, gated reference class
  rcf_unweighted  ablation: same, without the evidence hierarchy
  game_base_rate  theory only: outcome shares among same-game policies
  game_model      theory only: logistic regression of outcome on the incentive
                  margin and game, refit on each visible pool
  random_k        k random precedents through the same forecaster
  majority        climatology: always the pool base rate
  llm_rag         LLM with the retrieved precedents (optional)
  llm_closed      LLM with no retrieval (optional)

Metrics
  accuracy (all / answered only), coverage, macro-F1, multi-class Brier score,
  Brier skill score vs climatology, log loss, MAE on the net-outcome index,
  game accuracy, retrieval hit rate, citation validity and faithfulness.
  95% bootstrap CIs, plus a paired bootstrap of the Brier difference vs
  climatology, since n is small.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import config as C
from . import games as G
from .forecast import base_rate, reference_class_forecast
from .index import VectorStore
from .llm import PolicyLLM
from .retrieve import PolicyQuery, Precedent, candidate_pool, classify_game_knn, retrieve

CLS = C.OUTCOME_CLASSES


def _onehot(y: str) -> np.ndarray:
    return np.array([1.0 if c == y else 0.0 for c in CLS])


def _vec(p: dict[str, float]) -> np.ndarray:
    v = np.array([p[c] for c in CLS], dtype=float)
    return v / v.sum()


def _game_base_rate(df: pd.DataFrame, pool: np.ndarray, game: str) -> dict[str, float]:
    sub = pool & (df["game"] == game).to_numpy()
    if sub.sum() == 0:
        return base_rate(df, pool)
    # shrink same-game shares toward the pool base rate
    base = base_rate(df, pool)
    counts = df[sub]["outcome_direction"].value_counts()
    n = counts.sum()
    return {c: (C.PRIOR_STRENGTH * base[c] + counts.get(c, 0)) / (C.PRIOR_STRENGTH + n) for c in CLS}


def _game_features(margins, games) -> np.ndarray:
    return np.column_stack([np.asarray(margins, dtype=float)] +
                           [(np.asarray(games) == g).astype(float) for g in G.GAME_KEYS])


def _game_model(df: pd.DataFrame, pool: np.ndarray, game: str, m: float | None,
                base: dict[str, float], min_rows: int = 10) -> dict[str, float]:
    """Does the game model alone predict outcomes? Multinomial logistic
    regression of outcome on [margin, game one-hot], fit only on the visible
    pool. Falls back to the base rate when there is too little to fit."""
    fit = pool & df["game_scored"].to_numpy(dtype=bool)
    if m is None or fit.sum() < min_rows or df.loc[fit, "outcome_direction"].nunique() < 2:
        return dict(base)
    sub = df[fit]
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(_game_features(sub["margin"], sub["game"]), sub["outcome_direction"])
    pr = dict(zip(clf.classes_, clf.predict_proba(_game_features([m], [game]))[0]))
    return {c: 0.9 * pr.get(c, 0.0) + 0.1 * base[c] for c in CLS}


def _hit(precs: list[Precedent], target: dict) -> bool:
    for p in precs:
        r = p.record
        if p.above_threshold and r["game"] == target["game"] and (
            r["mechanism_type"] == target["mechanism_type"] or r["sub_sector"] == target["sub_sector"]
        ):
            return True
    return False


@dataclass
class BacktestConfig:
    mode: str = "cutoff"            # "cutoff" or "loo"
    k: int = C.TOP_K
    threshold: float | None = None
    use_llm: bool = False
    judge_claims: int = 2           # citation-faithfulness checks per target (LLM mode)
    min_pool: int = 5               # skip targets with too few visible precedents
    prior_strength: float = C.PRIOR_STRENGTH
    temperature: float = C.SIM_TEMPERATURE


def run_backtest(store: VectorStore, cfg: BacktestConfig, llm: PolicyLLM | None = None,
                 verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    df = store.df
    rng = np.random.default_rng(C.RANDOM_SEED)
    thr = store.threshold if cfg.threshold is None else cfg.threshold
    targets = df[df["has_outcome"]]
    if cfg.mode == "cutoff":
        targets = targets[targets["year_start"].notna()]

    rows = []
    for _, t in targets.iterrows():
        tr = t.to_dict()
        before = int(tr["year_start"]) if cfg.mode == "cutoff" else None
        pool = candidate_pool(df, before_year=before, exclude_family=tr["family"])
        opool = pool & df["has_outcome"].to_numpy()
        if opool.sum() < cfg.min_pool:
            continue

        q = PolicyQuery.from_record(tr)
        qvec = store.vectors[df.index.get_loc(t.name)]
        base = base_rate(df, opool)

        game_knn, _, knn_params = classify_game_knn(store, qvec, pool)
        if tr["game_scored"]:     # scored from design only -> as if scored at launch
            q.game, q.game_params = tr["game"], dict(tr["game_params"])
        else:
            q.game, q.game_params = game_knn, knn_params
        precs = retrieve(store, q, qvec, opool, k=cfg.k, threshold=thr)
        hp = dict(prior_strength=cfg.prior_strength, temperature=cfg.temperature)
        rcf = reference_class_forecast(precs, base, **hp)
        rcf_u = reference_class_forecast(
            retrieve(store, q, qvec, opool, k=cfg.k, threshold=thr), base,
            use_evidence_weights=False, **hp)

        # random-precedent baseline: same forecaster, precedents drawn at random
        idx = np.flatnonzero(opool)
        sims = store.similarities(qvec)
        rand_p = np.zeros(len(CLS))
        for _ in range(C.RANDOM_BASELINE_DRAWS):
            pick = rng.choice(idx, size=min(cfg.k, len(idx)), replace=False)
            rp = [Precedent(df.iloc[i].to_dict(), float(sims[i]), float(sims[i]), True) for i in pick]
            rand_p += _vec(reference_class_forecast(rp, base, gate=False, **hp).probs)
        rand_p /= C.RANDOM_BASELINE_DRAWS

        row = {
            "record_id": tr["record_id"], "policy_name": tr["policy_name"],
            "year": before, "pool_size": int(opool.sum()), "true": tr["outcome_direction"],
            "true_score": tr["outcome_score"], "true_game": tr["game"], "game_knn": game_knn,
            "margin": q.margin,
            "rcf_abstained": rcf.abstained, "rcf_n_eff": rcf.n_eff,
            "top_similarity": rcf.top_similarity, "retrieval_hit": _hit(precs, tr),
            "precedents": ";".join(p.record_id for p in precs if p.above_threshold),
        }
        dists = {
            "rcf": _vec(rcf.probs), "rcf_unweighted": _vec(rcf_u.probs),
            "game_base_rate": _vec(_game_base_rate(df, opool, q.game)),
            "game_model": _vec(_game_model(df, opool, q.game, q.margin, base)),
            "random_k": rand_p, "majority": _vec(base),
        }
        answered = {"rcf": not rcf.abstained, "rcf_unweighted": not rcf_u.abstained}

        if cfg.use_llm and llm is not None:
            choice = llm.score_game(q)
            row["game_llm"] = choice.game if choice else None
            if choice:
                q.game, q.game_params = choice.game, G.clean_params(choice.game, choice.param_dict())
            precs_l = retrieve(store, q, qvec, opool, k=cfg.k, threshold=thr)
            rcf_l = reference_class_forecast(precs_l, base, **hp)
            usable = [p for p in precs_l if p.above_threshold]
            if not rcf_l.abstained:
                lf = llm.predict(q, usable, G.describe(q.game, q.game_params), base, rcf_l.probs)
            else:
                lf = None
            if lf:
                dists["llm_rag"] = _vec(lf.probs())
                valid = {p.record_id for p in usable}
                cited = [r for c in lf.expected_effects for r in c.record_ids]
                row["citations"] = len(cited)
                row["citations_valid"] = sum(r in valid for r in cited)
                by_id = {p.record_id: p.record for p in usable}
                verdicts = []
                for c in lf.expected_effects[: cfg.judge_claims]:
                    recs = [by_id[r] for r in c.record_ids if r in by_id]
                    if recs:
                        j = llm.judge(c.claim, recs)
                        if j:
                            verdicts.append(j.supported)
                row["faithful_yes"] = verdicts.count("yes")
                row["faithful_partial"] = verdicts.count("partial")
                row["faithful_n"] = len(verdicts)
            else:
                dists["llm_rag"] = _vec(rcf_l.probs)   # abstain -> base rate
            answered["llm_rag"] = lf is not None
            lc = llm.predict_no_retrieval(q)
            dists["llm_closed"] = _vec(lc.probs()) if lc else _vec(base)
            answered["llm_closed"] = lc is not None

        for m, v in dists.items():
            for c, pv in zip(CLS, v):
                row[f"{m}__p_{c}"] = float(pv)
            row[f"{m}__pred"] = CLS[int(np.argmax(v))]
            row[f"{m}__answered"] = answered.get(m, True)
        rows.append(row)
        if verbose:
            print(f"  {tr['record_id'][:40]:40s} true={tr['outcome_direction']:16s} "
                  f"rcf={row['rcf__pred']:16s}{' (abstain)' if rcf.abstained else ''}")

    res = pd.DataFrame(rows)
    return res, summarise(res)


# ------------------------------------------------------------------ metrics
def _methods(res: pd.DataFrame) -> list[str]:
    return sorted({c.split("__")[0] for c in res.columns if c.endswith("__pred")},
                  key=lambda m: ["rcf", "llm_rag", "rcf_unweighted", "game_base_rate",
                                 "game_model", "random_k", "llm_closed", "majority"].index(m)
                  if m in ["rcf", "llm_rag", "rcf_unweighted", "game_base_rate",
                           "game_model", "random_k", "llm_closed", "majority"] else 99)


def _P(res: pd.DataFrame, m: str) -> np.ndarray:
    return res[[f"{m}__p_{c}" for c in CLS]].to_numpy()


def _boot_ci(values: np.ndarray, rng, stat=np.mean) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    bs = [stat(values[rng.integers(0, n, n)]) for _ in range(C.BOOTSTRAP_RESAMPLES)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarise(res: pd.DataFrame) -> dict:
    if res.empty:
        return {"n_targets": 0}
    rng = np.random.default_rng(C.RANDOM_SEED)
    Y = np.stack([_onehot(y) for y in res["true"]])
    brier_ref = ((_P(res, "majority") - Y) ** 2).sum(1)
    out = {"n_targets": int(len(res)), "methods": {}}
    for m in _methods(res):
        P = _P(res, m)
        pred = res[f"{m}__pred"].to_numpy()
        correct = (pred == res["true"].to_numpy()).astype(float)
        ans = res[f"{m}__answered"].to_numpy(dtype=bool)
        brier = ((P - Y) ** 2).sum(1)
        ll = -np.log(np.clip((P * Y).sum(1), 1e-9, None))
        exp_score = P @ np.array([C.OUTCOME_SCORE[c] for c in CLS])
        diff = brier - brier_ref
        # direction accuracy (proposal metric): up / flat-or-mixed / down
        P3 = np.stack([P[:, 0], P[:, 1] + P[:, 2], P[:, 3]], 1)
        Y3 = np.stack([Y[:, 0], Y[:, 1] + Y[:, 2], Y[:, 3]], 1)
        dir_correct = (P3.argmax(1) == Y3.argmax(1)).astype(float)
        out["methods"][m] = {
            "accuracy": float(correct.mean()),
            "accuracy_ci95": _boot_ci(correct, rng),
            "direction_accuracy": float(dir_correct.mean()),
            "direction_accuracy_ci95": _boot_ci(dir_correct, rng),
            "coverage": float(ans.mean()),
            "accuracy_answered": float(correct[ans].mean()) if ans.any() else None,
            "macro_f1": float(f1_score(res["true"], pred, labels=CLS, average="macro", zero_division=0)),
            "brier": float(brier.mean()),
            "brier_ci95": _boot_ci(brier, rng),
            "brier_skill_vs_climatology": float(1 - brier.mean() / brier_ref.mean()),
            "brier_diff_vs_climatology_ci95": _boot_ci(diff, rng),
            "log_loss": float(ll.mean()),
            "mae_net_outcome": float(np.abs(exp_score - res["true_score"].to_numpy()).mean()),
        }
    out["game_accuracy_knn"] = float((res["game_knn"] == res["true_game"]).mean())
    if "game_llm" in res:
        out["game_accuracy_llm"] = float((res["game_llm"] == res["true_game"]).mean())
    out["retrieval_hit_rate"] = float(res["retrieval_hit"].mean())
    out["mean_top_similarity"] = float(res["top_similarity"].mean())
    if "citations" in res and res["citations"].sum() > 0:
        out["citation_validity"] = float(res["citations_valid"].sum() / res["citations"].sum())
    if "faithful_n" in res and res["faithful_n"].sum() > 0:
        n = res["faithful_n"].sum()
        out["citation_faithfulness_yes"] = float(res["faithful_yes"].sum() / n)
        out["citation_faithfulness_yes_or_partial"] = float(
            (res["faithful_yes"].sum() + res["faithful_partial"].sum()) / n)
    out["class_distribution"] = res["true"].value_counts().to_dict()
    return out


def coverage_curve(store: VectorStore, mode: str = "cutoff",
                   quantiles=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) -> pd.DataFrame:
    """Selective-prediction trade-off: a stricter relevance gate answers fewer
    policies. Does it answer them more accurately?"""
    off = store.pairwise_offdiag()
    rows = []
    for qt in quantiles:
        thr = float(np.quantile(off, qt))
        res, _ = run_backtest(store, BacktestConfig(mode=mode, threshold=thr), verbose=False)
        ans = res["rcf__answered"].astype(bool)
        corr = res["rcf__pred"] == res["true"]
        rows.append({"quantile": qt, "threshold": thr, "coverage": ans.mean(),
                     "accuracy_answered": corr[ans].mean() if ans.any() else np.nan,
                     "majority_accuracy_same_subset": (res.loc[ans, "majority__pred"] ==
                                                       res.loc[ans, "true"]).mean() if ans.any() else np.nan,
                     "n_answered": int(ans.sum())})
    return pd.DataFrame(rows)


def sensitivity(store: VectorStore, mode: str = "loo") -> pd.DataFrame:
    """Robustness of the reference-class forecaster to its hyperparameters.

    Reported, not used for tuning: the headline numbers use the a-priori
    values in config.py, so there is no selection on the test set."""
    import itertools

    rows = []
    for a, t, k in itertools.product([0.5, 2.0, 5.0], [0.02, 0.05, 0.15], [5, 8, 12]):
        _, m = run_backtest(store, BacktestConfig(mode=mode, k=k, prior_strength=a, temperature=t),
                            verbose=False)
        s = m["methods"]["rcf"]
        rows.append({"prior_strength": a, "temperature": t, "k": k,
                     "accuracy": s["accuracy"], "direction_accuracy": s["direction_accuracy"],
                     "brier": s["brier"], "brier_skill": s["brier_skill_vs_climatology"],
                     "coverage": s["coverage"]})
    return pd.DataFrame(rows)


def save(res: pd.DataFrame, metrics: dict, mode: str, curve: pd.DataFrame | None = None):
    out = C.ARTIFACTS / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{mode}_predictions.csv", index=False)
    (out / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if curve is not None:
        curve.to_csv(out / f"{mode}_coverage.csv", index=False)
