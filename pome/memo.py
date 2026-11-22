"""Render a prediction as a decision memo: the management deliverable."""
from __future__ import annotations

from datetime import date

from . import config as C
from . import games as G
from .pipeline import PredictionResult

LABEL = {"improved": "Improved", "mixed": "Mixed", "no_clear_effect": "No clear effect",
         "worsened": "Worsened"}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _year(r: dict) -> str:
    y = r.get("year_start")
    return "n/a" if y is None or str(y) == "<NA>" else str(int(y))


def render_memo(res: PredictionResult) -> str:
    q, fc, lf = res.query, res.forecast, res.llm_forecast
    game = G.GAMES.get(res.game)
    gp = G.clean_params(res.game, res.game_params) if game else None
    probs = res.final_probs
    used = [p for p in res.precedents if p.above_threshold and p.record.get("has_outcome")]
    L: list[str] = []
    add = L.append

    add(f"# Decision memo: {q.policy_name}")
    add(f"*Policy Outcome Predictor · {date.today():%d %b %Y} · decision support, not a decision*\n")

    # ---- 1. bottom line
    add("## 1. Bottom line")
    if fc.abstained:
        add(f"**Insufficient evidence.** {fc.note} The corpus base rate is shown below "
            "for orientation only; it shouldn't be read as a forecast for this policy.\n")
    else:
        src = "evidence-weighted reference class, adjusted by LLM analysis" if lf else \
              "evidence-weighted reference class"
        add(f"**Most likely outcome: {LABEL[res.final_label]}** "
            f"({_pct(probs[res.final_label])}), confidence **{fc.confidence}** ({src}).\n")
    add("| Outcome | Probability | Corpus base rate |\n|---|---|---|")
    for c in C.OUTCOME_CLASSES:
        add(f"| {LABEL[c]} | {_pct(probs[c])} | {_pct(fc.base_rate.get(c, 0))} |")
    net = sum(probs[c] * C.OUTCOME_SCORE[c] for c in C.OUTCOME_CLASSES)
    add(f"\nNet-outcome index: **{net:+.2f}** (on a scale from -1, certain to worsen, "
        f"to +1, certain to improve). Reference class: {fc.n_precedents} precedents, "
        f"effective sample size {fc.n_eff:.1f}, best match similarity {fc.top_similarity:.2f} "
        f"(relevance gate {res.threshold:.2f}).\n")
    if lf:
        add(f"**Why the forecast moves from the base rate:** {lf.adjustment_from_base_rate}\n")

    # ---- 2. game model
    if game:
        add(f"## 2. Game model: {game.name}")
        add(f"*Assigned by: {res.game_source}.* {res.game_rationale}\n")
        add(f"**Players:** {game.players}\n\n**Strategic question:** {game.question}\n")
        add(game.payoff_table + "\n")
        if gp:
            add("| Parameter | Meaning | Score (0-1) | Why |\n|---|---|---|---|")
            for prm in game.params:
                add(f"| {prm.sym} | {prm.name} | {gp[prm.sym]:.2f} | "
                    f"{res.param_reasons.get(prm.sym, '-')} |")
            b = G.binding_term(res.game, gp)
            add(f"\n**Incentive margin:** `{G.equation(res.game, gp)}` "
                f"(m > 0: incentives favour the policy working).\n")
            add(f"{G.regime(res.game, gp)} {game.threshold_note}\n")
            add(f"**Binding constraint:** {b} ({game.param(b).name}), the parameter whose "
                f"improvement would raise the margin most.\n")
        else:
            add("*Parameters not scored (no LLM and no scored precedents of this game).*\n")
        add(f"*Theory: {game.theory_ref}*\n")

    # ---- 3. evidence
    add("## 3. Reference class (precedents)")
    if not res.precedents:
        add("No precedents in the candidate pool.\n")
    else:
        add("| # | Precedent | Year | Similarity | Weight | Outcome | Evidence | Shared | Differs |")
        add("|---|---|---|---|---|---|---|---|---|")
        for i, p in enumerate(res.precedents, 1):
            r = p.record
            gate = "" if p.above_threshold else " *(below gate)*"
            add(f"| {i} | [{r['policy_name']}]({r['source_url']}){gate} | {_year(r)} | "
                f"{p.similarity:.2f} | {p.weight:.2f} | {LABEL.get(r['outcome_direction'], r['outcome_direction'])} | "
                f"{r['evidence_type']}/{r['evidence_quality']} | {'; '.join(p.shared) or '-'} | "
                f"{'; '.join(d for d in p.differs if not d.startswith('LLM')) or '-'} |")
        add("")
        for p in used[:5]:
            r = p.record
            add(f"- **{r['policy_name']}** `[{r['record_id']}]`: {r['outcome_summary']} "
                f"*Side effects:* {r['side_effects']}")
        add("")

    # ---- 4. what separated success from failure
    good = [p for p in used if p.record["outcome_direction"] == "improved"]
    bad = [p for p in used if p.record["outcome_direction"] in ("worsened", "no_clear_effect")]
    if good and bad:
        add("## 4. What separated success from failure among the precedents")
        add("**Improved:**")
        L.extend(f"- {p.record['policy_name']}: {p.record['description']} "
                 f"({p.record['mechanism_type']})" for p in good[:3])
        add("\n**Worsened or no clear effect:**")
        L.extend(f"- {p.record['policy_name']}: {p.record['description']} "
                 f"({p.record['mechanism_type']})" for p in bad[:3])
        add("")

    # ---- 5. risk register
    add("## 5. Risk register")
    if lf and lf.risks:
        add("| Risk | Likelihood | Mitigation | Evidence |\n|---|---|---|---|")
        for r in lf.risks:
            add(f"| {r.risk} | {r.likelihood} | {r.mitigation} | {', '.join(r.record_ids) or '-'} |")
    else:
        add("| Risk observed in a precedent | Precedent outcome | Evidence | Source |\n|---|---|---|---|")
        for p in sorted(used, key=lambda p: -p.weight)[:6]:
            r = p.record
            if r["side_effects"] and not r["side_effects"].lower().startswith("not verified"):
                add(f"| {r['side_effects']} | {LABEL.get(r['outcome_direction'])} | "
                    f"{r['evidence_type']}/{r['evidence_quality']} | `{r['record_id']}` |")
    add("")

    # ---- 6. levers
    add("## 6. Design levers to consider")
    if lf and lf.design_recommendations:
        L.extend(f"- {d}" for d in lf.design_recommendations)
    if game and gp:
        add("\n*Levers from the game model, largest margin gain first:*")
        for sym, gain in sorted(G.gains(res.game, gp).items(), key=lambda x: -x[1]):
            if gain > 0.01:
                add(f"- **{sym}** ({game.param(sym).name}, +{gain:.2f} to m): {game.param(sym).lever}")
    add("")

    if lf and lf.expected_effects:
        add("## 7. Expected effects (cited)")
        L.extend(f"- {c.claim} `[{', '.join(c.record_ids)}]`" for c in lf.expected_effects)
        add("\n**Key uncertainties:**")
        L.extend(f"- {u}" for u in lf.key_uncertainties)
        add("")
    if res.invalid_citations:
        add(f"> Warning: the LLM cited record_ids outside the retrieved evidence: "
            f"{', '.join(res.invalid_citations)}. Treat those claims as unsupported.\n")

    add("## Limitations")
    add("- The corpus holds 57 records; probabilities are coarse and should be read as ranges.\n"
        "- Outcome labels summarise heterogeneous metrics; 'mixed' is the most common label.\n"
        "- Precedents are associations, not causal estimates for this policy's context.\n"
        "- Game assignments and parameter scores are the project's modelling judgements "
        "(LLM-drafted, human-reviewed), not claims made by the sources.")
    return "\n".join(L)
