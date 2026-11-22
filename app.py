"""Policy Outcome Predictor: Streamlit front end.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pome import config as C
from pome.analytics import run_all
from pome import game_plots as GP
from pome import games as G
from pome.llm import PolicyLLM
from pome.memo import LABEL, render_memo
from pome.pipeline import PolicyPredictor
from pome.retrieve import PolicyQuery

st.set_page_config(page_title="Policy Outcome Predictor", layout="wide")
COLORS = {"improved": "#1b7f5b", "mixed": "#c9a227", "no_clear_effect": "#8a8f98", "worsened": "#b23a3a"}


@st.cache_resource(show_spinner="Loading vector index ...")
def get_predictor(use_llm: bool) -> PolicyPredictor:
    return PolicyPredictor(use_llm=use_llm)


@st.cache_data
def get_analytics():
    pred = get_predictor(False)
    return run_all(pred.store.df)


llm_ok = PolicyLLM.available()
with st.sidebar:
    st.title("Policy Outcome Predictor")
    st.caption("Retrieval-augmented reference-class forecasting of Indian policy outcomes.")
    use_llm = st.toggle("Use LLM (inside view)", value=llm_ok, disabled=not llm_ok,
                        help="Needs GROQ_API_KEY. Without it the evidence-weighted "
                             "reference-class forecast runs fully offline.")
    llm_rerank = st.toggle("LLM re-ranking of precedents", value=False, disabled=not use_llm)
    k = st.slider("Precedents to retrieve (k)", 3, 15, C.TOP_K)
    pred = get_predictor(use_llm)
    st.caption(f"Embeddings: {pred.store.embedder.label} · gate {pred.store.threshold:.2f} · "
               f"{len(pred.store.df)} records")

tab_fc, tab_bt, tab_an, tab_data, tab_method = st.tabs(
    ["Forecast a policy", "Backtest", "Evidence analytics", "Data", "Method"])

# ======================================================================== forecast
with tab_fc:
    df = pred.store.df
    examples = {
        "(blank)": {},
        "Free farm power (Punjab)": dict(
            name="Free electricity for agricultural pumps",
            desc="State supplies unmetered, free electricity to all agricultural tubewells",
            goal="Lower irrigation costs and support farm incomes", target="Farmers with electric pump sets",
            sector="energy_water_environment", mech="subsidy", level="state", state="Punjab"),
        "Cash for girls' secondary schooling": dict(
            name="Conditional cash transfer for girls completing Class 10",
            desc="Rs 10,000 paid by DBT to girls who pass Class 10 board exams in government schools",
            goal="Raise girls' secondary completion", target="Girls in government schools",
            sector="social_welfare", mech="cash_transfer", level="state", state="Bihar"),
        "Carbon tax on coal power": dict(
            name="Carbon levy on coal-fired electricity",
            desc="Per-tonne CO2 levy on thermal power generators, revenue recycled to renewables",
            goal="Reduce emissions intensity of electricity", target="Thermal power plants",
            sector="energy_water_environment", mech="tax", level="central", state=""),
    }
    ex = examples[st.selectbox("Start from an example", list(examples))]
    with st.form("policy"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Policy name", ex.get("name", ""))
        target = c2.text_input("Target group", ex.get("target", ""))
        desc = st.text_area("Design: what does the policy do?", ex.get("desc", ""), height=90)
        goal = st.text_input("Stated goal", ex.get("goal", ""))
        c3, c4, c5, c6 = st.columns(4)
        sectors = [""] + sorted(df.sector.unique())
        mechs = [""] + sorted(df.mechanism_type.unique())
        sector = c3.selectbox("Sector", sectors, index=sectors.index(ex.get("sector", "")))
        mech = c4.selectbox("Instrument", mechs, index=mechs.index(ex.get("mech", "")))
        level = c5.selectbox("Level", ["", "central", "state"],
                             index=["", "central", "state"].index(ex.get("level", "")))
        state = c6.text_input("State", ex.get("state", ""))
        game_opt = st.selectbox("Game model", ["(auto: LLM or nearest precedents)"] + C.GAMES,
                                format_func=lambda g: G.GAMES[g].name if g in G.GAMES else g)
        hard = st.multiselect("Hard filters (restrict precedents to the same ...)",
                              ["sector", "mechanism_type", "level"])
        go_btn = st.form_submit_button("Forecast", type="primary")

    if go_btn:
        if not name or not desc:
            st.error("Give at least a name and a design description.")
            st.stop()
        q = PolicyQuery(policy_name=name, description=desc, stated_goal=goal, target_group=target,
                        sector=sector, mechanism_type=mech, level=level, state=state,
                        game=None if game_opt.startswith("(") else game_opt)
        filters = {f: getattr(q, f) for f in hard if getattr(q, f)}
        with st.spinner("Retrieving precedents and forecasting ..."):
            res = pred.predict(q, hard_filters=filters, k=k, use_llm=use_llm, llm_rerank=llm_rerank)
        fc = res.forecast
        probs = res.final_probs

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Most likely outcome", "Insufficient evidence" if fc.abstained else LABEL[res.final_label],
                  None if fc.abstained else f"{probs[res.final_label]:.0%}")
        m2.metric("Confidence", fc.confidence)
        m3.metric("Effective precedents", f"{fc.n_eff:.1f}", f"{fc.n_precedents} above gate",
                  delta_color="off")
        net = sum(probs[c] * C.OUTCOME_SCORE[c] for c in C.OUTCOME_CLASSES)
        m4.metric("Net-outcome index", f"{net:+.2f}", help="-1 = certain to worsen, +1 = certain to improve")

        cA, cB = st.columns([3, 2])
        with cA:
            bars = go.Figure()
            series = [("Corpus base rate", fc.base_rate), ("Reference class", fc.probs)]
            if res.llm_forecast:
                series.append(("Final (LLM-adjusted)", res.llm_forecast.probs()))
            for nm, p in series:
                bars.add_bar(name=nm, x=[LABEL[c] for c in C.OUTCOME_CLASSES],
                             y=[p[c] for c in C.OUTCOME_CLASSES])
            bars.update_layout(barmode="group", yaxis_tickformat=".0%", height=320,
                               margin=dict(t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(bars, width="stretch")
            st.caption(fc.note)
        with cB:
            game = G.GAMES[res.game]
            gp = G.clean_params(res.game, res.game_params)
            st.subheader(f"Game: {game.name}")
            st.caption(f"Assigned by {res.game_source}. {res.game_rationale}")
            st.write(game.question)
            if gp:
                m = res.margin
                st.metric("Incentive margin m", f"{m:+.2f}",
                          "incentives favour success" if m >= 0 else "incentives work against it",
                          delta_color="normal" if m >= 0 else "inverse",
                          help=f"{game.formula}; m > 0 means the key players gain by acting as the policy intends")
                st.code(G.equation(res.game, gp), language=None)
                b = res.binding_term
                st.write(f"{G.regime(res.game, gp)}  \n**Binding constraint:** {b} "
                         f"({game.param(b).name}). {game.param(b).lever}")
            else:
                st.caption("Parameters not scored for this policy.")
            with st.expander("Payoffs and theory"):
                st.markdown(f"**Players:** {game.players}\n\n" + game.payoff_table +
                            f"\n\n{game.threshold_note}\n\n*{game.theory_ref}*")

        if gp:
            st.subheader(f"Game model graphs: {game.name}")
            g1, g2 = st.columns([3, 2])
            with g1:
                st.plotly_chart(GP.payoff_chart(res.game, gp), width="stretch")
                st.caption({
                    "principal_agent": "The agent complies where the green line (reward for complying) is above "
                                       "the red one (expected gain from deviating). They cross at p*, the "
                                       "detection rate the policy needs.",
                    "collective_action": "Above zero, a user gains by cooperating; below zero, by defecting. The dot "
                                         "is the expected share cooperating at launch; the arrow shows whether "
                                         "cooperation spreads or unravels from there.",
                    "commitment": "People respond where the green line (expected payoff of responding) is above "
                                  "zero (waiting). They cross at κ*, the credibility the policy needs.",
                }[res.game])
            with g2:
                st.plotly_chart(GP.lever_chart(res.game, gp), width="stretch")
                st.caption("Each bar: how much the margin m would rise if that parameter were moved to its "
                           "best value, others unchanged. The dark bar is the binding constraint.")
            with st.expander("Parameter scores and where this policy sits among past policies"):
                st.plotly_chart(GP.params_chart(res.game, gp, res.param_reasons), width="stretch")
                st.caption("Green: higher helps the policy. Red: higher hurts it. Hover for the reason.")
                st.plotly_chart(GP.margin_outcome_chart(pred.store.df, res.margin), width="stretch")

        st.subheader("Precedents")
        rows = [{
            "Policy": p.record["policy_name"], "Similarity": round(p.similarity, 3),
            "Weight": round(p.weight, 3), "Above gate": p.above_threshold,
            "Outcome": LABEL.get(p.record["outcome_direction"], p.record["outcome_direction"]),
            "Evidence": f"{p.record['evidence_type']} / {p.record['evidence_quality']}",
            "Game": p.record["game"],
            "Margin": None if pd.isna(p.record["margin"]) else round(float(p.record["margin"]), 2),
            "Shared": "; ".join(p.shared),
            "Implication": p.record["outcome_summary"], "Source": p.record["source_url"],
        } for p in res.precedents]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"Source": st.column_config.LinkColumn("Source")})

        memo = render_memo(res)
        with st.expander("Decision memo", expanded=True):
            st.markdown(memo)
        st.download_button("Download memo (.md)", memo, file_name="decision_memo.md")
        if use_llm and pred.llm and pred.llm.errors:
            st.warning(f"LLM issues (fell back to reference class where needed): {pred.llm.errors[-1]}")

# ======================================================================== backtest
with tab_bt:
    st.subheader("Backtest: hide the outcome, predict, compare")
    st.write("Each past policy is held out together with every row about the same policy; "
             "in *cutoff* mode only policies that started earlier are visible. Run "
             "`python scripts/backtest.py` to refresh these results.")
    bt_dir = C.ARTIFACTS / "backtest"
    mode = st.radio("Mode", ["cutoff", "loo"], horizontal=True,
                    format_func=lambda m: "Date cutoff (strict)" if m == "cutoff" else "Leave-one-family-out")
    mpath = bt_dir / f"{mode}_metrics.json"
    if not mpath.exists():
        st.info("No backtest results yet. Run `python scripts/backtest.py`.")
    else:
        m = json.loads(mpath.read_text())
        tbl = pd.DataFrame([{
            "Method": k, "Accuracy": v["accuracy"],
            "95% CI": f"{v['accuracy_ci95'][0]:.2f} to {v['accuracy_ci95'][1]:.2f}",
            "Coverage": v["coverage"], "Acc. when answering": v["accuracy_answered"],
            "Macro-F1": v["macro_f1"], "Brier": v["brier"],
            "Brier skill vs base rate": v["brier_skill_vs_climatology"], "Log loss": v["log_loss"],
            "MAE net outcome": v["mae_net_outcome"],
        } for k, v in m["methods"].items()])
        st.dataframe(tbl.style.format(precision=3), width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Targets", m["n_targets"])
        c2.metric("Game accuracy (kNN)", f"{m['game_accuracy_knn']:.0%}")
        c3.metric("Retrieval hit rate", f"{m['retrieval_hit_rate']:.0%}")
        for key, lbl in [("game_accuracy_llm", "Game accuracy (LLM)"),
                         ("citation_validity", "Citation validity"),
                         ("citation_faithfulness_yes_or_partial", "Citation faithfulness")]:
            if key in m:
                st.metric(lbl, f"{m[key]:.0%}")
        cpath = bt_dir / f"{mode}_coverage.csv"
        if cpath.exists():
            cv = pd.read_csv(cpath)
            fig = go.Figure()
            fig.add_scatter(x=cv["coverage"], y=cv["accuracy_answered"], mode="lines+markers",
                            name="Reference class", text=cv["quantile"])
            fig.add_scatter(x=cv["coverage"], y=cv["majority_accuracy_same_subset"],
                            mode="lines+markers", name="Base rate, same subset", line=dict(dash="dot"))
            fig.update_layout(title="Selective prediction: stricter gate, fewer but better answers?",
                              xaxis_title="Coverage (share of policies answered)",
                              yaxis_title="Accuracy on answered", height=360,
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        ppath = bt_dir / f"{mode}_predictions.csv"
        if ppath.exists():
            with st.expander("Per-policy predictions"):
                st.dataframe(pd.read_csv(ppath)[["record_id", "year", "pool_size", "true", "rcf__pred",
                                                 "rcf_abstained", "rcf_n_eff", "top_similarity",
                                                 "true_game", "game_knn", "margin", "precedents"]],
                             width="stretch", hide_index=True)

# ======================================================================== analytics
with tab_an:
    a = get_analytics()
    st.subheader("What the record says")
    tests = pd.DataFrame([{"Factor": k, "chi2": t["chi2"], "p-value": t["p_value"],
                           "Cramer's V": t["cramers_v"], "n": t["n"]} for k, t in a["tests"].items()])
    st.write("Association between each factor and the outcome (permutation chi-square):")
    st.dataframe(tests.style.format(precision=3), hide_index=True)
    s = a["evidence_vs_outcome_spearman"]
    st.write(f"Evidence strength vs net outcome: Spearman rho = **{s['rho']:+.2f}** "
             f"(p = {s['p_value']:.3f}). Negative means more rigorous evaluations give harsher verdicts.")
    mt = a["margin_vs_outcome_spearman"]
    st.write(f"Game model: incentive margin vs net outcome, Spearman rho = **{mt['rho']:+.2f}** "
             f"(p = {mt['p_value']:.3f}, n = {mt['n']}). Positive means policies whose design gives "
             f"the key players an incentive to comply tend to turn out better.")
    st.plotly_chart(GP.margin_outcome_chart(pred.store.df), width="stretch")
    st.caption("Each point is a past policy. Colour is its real outcome, shape is its game. The vertical "
               "ticks are the mean margin per outcome: if the game model works, they move left as outcomes "
               "get worse.")
    choice = st.selectbox("Break down by", ["by_game", "by_mechanism", "by_sector", "by_evidence", "by_level"],
                          format_func=lambda x: x.replace("by_", "").replace("_", " "))
    t = a[choice]
    col = t.columns[0]
    d = pred.store.df[pred.store.df.has_outcome]
    ct = pd.crosstab(d[col], d["outcome_direction"]).reindex(columns=C.OUTCOME_CLASSES, fill_value=0)
    ct = ct.loc[t[col]]
    fig = px.bar(ct, orientation="h", color_discrete_map=COLORS,
                 labels={"value": "policies", col: ""}, height=60 + 32 * len(ct))
    fig.update_layout(legend_title_text="", yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(t.style.format({c: "{:.2f}" for c in t.columns if t[c].dtype == float}),
                 hide_index=True, width="stretch")
    st.caption("Small cells: treat differences as hypotheses, not findings.")

# ======================================================================== data
with tab_data:
    st.dataframe(pred.store.df.drop(columns=["design_text", "game_params"]), width="stretch",
                 hide_index=True, column_config={"source_url": st.column_config.LinkColumn()})
    if C.GAME_PARAMS_PATH.exists():
        st.subheader("Game-model scores (game_params.csv)")
        st.caption("LLM-drafted from design fields only; 'reviewed' marks rows a person has checked.")
        st.dataframe(pd.read_csv(C.GAME_PARAMS_PATH), width="stretch", hide_index=True)

# ======================================================================== method
with tab_method:
    st.markdown(open(C.ROOT / "docs" / "METHOD.md", encoding="utf-8").read()
                if (C.ROOT / "docs" / "METHOD.md").exists() else "See README.md")
