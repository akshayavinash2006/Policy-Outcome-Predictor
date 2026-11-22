"""Descriptive and inferential analysis of the evidence base.

These are the "economics" findings of the project in their own right: what
does the Indian policy record say about which incentive structures and
instruments tend to work, and how does the type of evidence relate to the
verdict? Every test is a permutation test, because cell counts are small and
asymptotic chi-square p-values would be unreliable."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _chi2(tab: np.ndarray) -> float:
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp)))


def permutation_chi2(x: pd.Series, y: pd.Series, n_perm: int = 5000, seed: int = C.RANDOM_SEED) -> dict:
    tab = pd.crosstab(x, y)
    obs = _chi2(tab.to_numpy(float))
    rng = np.random.default_rng(seed)
    xc, xu = pd.factorize(x.to_numpy())
    yc, yu = pd.factorize(y.to_numpy())
    r_, k_ = len(xu), len(yu)
    ge = 0
    for _ in range(n_perm):
        t = np.bincount(xc * k_ + rng.permutation(yc), minlength=r_ * k_).reshape(r_, k_)
        ge += _chi2(t.astype(float)) >= obs - 1e-12
    n = tab.to_numpy().sum()
    r, k = tab.shape
    v = np.sqrt(obs / (n * max(1, min(r, k) - 1)))
    return {"chi2": obs, "p_value": (ge + 1) / (n_perm + 1), "cramers_v": float(v),
            "n": int(n), "table": tab}


def success_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Share of policies rated 'improved' (and 'worsened') by group, with Wilson CIs."""
    d = df[df["has_outcome"]]
    rows = []
    for g, sub in d.groupby(by):
        n = len(sub)
        k_imp = int((sub["outcome_direction"] == "improved").sum())
        k_wor = int((sub["outcome_direction"] == "worsened").sum())
        lo, hi = wilson(k_imp, n)
        rows.append({by: g, "n": n, "improved": k_imp, "worsened": k_wor,
                     "share_improved": k_imp / n, "improved_ci_low": lo, "improved_ci_high": hi,
                     "share_worsened": k_wor / n,
                     "mean_net_outcome": float(sub["outcome_score"].mean())})
    return pd.DataFrame(rows).sort_values("mean_net_outcome", ascending=False)


def margin_test(d: pd.DataFrame, n_perm: int = 5000) -> dict:
    """Headline test of the game model: does a higher incentive margin (scored
    from design only) go with better outcomes? Spearman rho, permutation p."""
    s = d[d["game_scored"].astype(bool)]
    if len(s) < 8:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(len(s)),
                "mean_score_m_pos": float("nan"), "mean_score_m_neg": float("nan")}
    rx = s["margin"].rank().to_numpy()
    ry = s["outcome_score"].rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(C.RANDOM_SEED)
    perm = [np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)]
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1)
    pos, neg = s[s["margin"] >= 0], s[s["margin"] < 0]
    return {"rho": rho, "p_value": float(p), "n": int(len(s)),
            "n_m_pos": int(len(pos)), "n_m_neg": int(len(neg)),
            "mean_score_m_pos": float(pos["outcome_score"].mean()) if len(pos) else float("nan"),
            "mean_score_m_neg": float(neg["outcome_score"].mean()) if len(neg) else float("nan")}


def run_all(df: pd.DataFrame) -> dict:
    d = df[df["has_outcome"]]
    tests = {}
    for col in ["game", "mechanism_type", "sector", "evidence_type", "level"]:
        tests[col] = permutation_chi2(d[col], d["outcome_direction"])
    # Does stronger evidence deliver harsher verdicts? (net outcome vs evidence weight)
    rho = d[["evidence_weight", "outcome_score"]].corr(method="spearman").iloc[0, 1]
    rng = np.random.default_rng(C.RANDOM_SEED)
    rx = d["evidence_weight"].rank().to_numpy()
    ry = d["outcome_score"].rank().to_numpy()
    perm = [np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(5000)]
    p_rho = (np.sum(np.abs(perm) >= abs(rho)) + 1) / 5001
    return {
        "tests": tests,
        "evidence_vs_outcome_spearman": {"rho": float(rho), "p_value": float(p_rho)},
        "margin_vs_outcome_spearman": margin_test(d),
        "by_game": success_table(df, "game"),
        "by_mechanism": success_table(df, "mechanism_type"),
        "by_sector": success_table(df, "sector"),
        "by_evidence": success_table(df, "evidence_type"),
        "by_level": success_table(df, "level"),
    }


def to_markdown(a: dict, df: pd.DataFrame) -> str:
    L = ["# Evidence-base analytics", "",
         f"Records: {len(df)} · with a known outcome: {int(df['has_outcome'].sum())} · "
         f"policy families: {df['family'].nunique()}", ""]
    L.append("## Is the outcome associated with ...? (permutation chi-square, 5,000 permutations)")
    L.append("| Factor | chi2 | p-value | Cramer's V | n |\n|---|---|---|---|---|")
    for k, t in a["tests"].items():
        L.append(f"| {k} | {t['chi2']:.2f} | {t['p_value']:.3f} | {t['cramers_v']:.2f} | {t['n']} |")
    s = a["evidence_vs_outcome_spearman"]
    L += ["", f"Evidence strength vs net outcome: Spearman rho = {s['rho']:+.2f} "
          f"(permutation p = {s['p_value']:.3f}). A negative rho means more rigorous "
          "evaluations tend to give harsher verdicts.", ""]
    mt = a["margin_vs_outcome_spearman"]
    L += ["## Game model: does the incentive margin predict the outcome?",
          f"Spearman rho between incentive margin m (scored from design only) and net outcome: "
          f"**{mt['rho']:+.2f}** (permutation p = {mt['p_value']:.3f}, n = {mt['n']}). "
          f"Mean net outcome when m >= 0: {mt['mean_score_m_pos']:+.2f} "
          f"(n = {mt.get('n_m_pos', 0)}); when m < 0: {mt['mean_score_m_neg']:+.2f} "
          f"(n = {mt.get('n_m_neg', 0)}).", ""]
    for key, title in [("by_game", "game model"), ("by_mechanism", "policy instrument"),
                       ("by_sector", "sector"), ("by_evidence", "evidence type"),
                       ("by_level", "level of government")]:
        t = a[key]
        col = t.columns[0]
        L.append(f"## Outcomes by {title}")
        L.append("| Group | n | improved | worsened | share improved (95% CI) | net outcome |")
        L.append("|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            L.append(f"| {r[col]} | {r['n']} | {r['improved']} | {r['worsened']} | "
                     f"{r['share_improved']:.0%} ({r['improved_ci_low']:.0%} to {r['improved_ci_high']:.0%}) | "
                     f"{r['mean_net_outcome']:+.2f} |")
        L.append("")
    L.append("*Small cells: treat group differences as hypotheses, not findings.*")
    return "\n".join(L)
