"""Plotly charts for the game model.

Each game gets one "payoff diagram" that turns its inequality into a picture:

  G1 Principal-agent    payoff of complying (w) vs deviating (g - p*F) as the
                        detection probability p rises; they cross at p*.
  G2 Collective action  the gain from cooperating, t - g + k*x, against the
                        share x of others cooperating; it crosses zero at the
                        tipping point x*, and the arrow shows which way
                        behaviour drifts from the starting share x0.
  G3 Commitment         payoff of responding (kappa*b - c) vs waiting (0) as
                        credibility kappa rises; they cross at kappa*.

The current policy is marked on each curve. Green regions are where the
intended behaviour pays, red where it doesn't. lever_chart() shows how much
the margin would rise if each parameter were fixed (the binding constraint
is the longest bar), and margin_outcome_chart() plots every record's margin
against its real outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import games as G

GREEN, RED, GREY, INK = "#1b7f5b", "#b23a3a", "#8a8f98", "#2b2f36"
GREEN_BG, RED_BG = "rgba(27,127,91,0.10)", "rgba(178,58,58,0.10)"
OUTCOME_ORDER = ["improved", "mixed", "no_clear_effect", "worsened"]
OUTCOME_LABEL = {"improved": "Improved", "mixed": "Mixed", "no_clear_effect": "No clear effect",
                 "worsened": "Worsened"}
OUTCOME_COLOR = {"improved": GREEN, "mixed": "#c9a227", "no_clear_effect": GREY, "worsened": RED}


def _layout(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height,
        margin=dict(t=60, b=50, l=55, r=20), legend=dict(orientation="h", y=-0.22),
        xaxis=dict(title=xtitle, range=[0, 1], tickformat=".0%" if "share" in xtitle.lower() else ".1f"),
        yaxis=dict(title=ytitle, zeroline=True, zerolinecolor="#999", zerolinewidth=1),
        plot_bgcolor="white", hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def _regions(fig: go.Figure, cross: float | None, good_above: bool, good_label: str, bad_label: str):
    """Shade [0, cross] and [cross, 1]; good_above says which side is green."""
    def band(x0, x1, good):
        if x1 - x0 <= 1e-6:
            return
        fig.add_vrect(x0=x0, x1=x1, fillcolor=GREEN_BG if good else RED_BG, line_width=0, layer="below",
                      annotation_text=good_label if good else bad_label,
                      annotation_position="top left", annotation_font_size=11,
                      annotation_font_color=GREEN if good else RED)
    if cross is None:
        return
    c = min(1.0, max(0.0, cross))
    band(0.0, c, not good_above)
    band(c, 1.0, good_above)


def _threshold_line(fig: go.Figure, x: float | None, label: str):
    if x is not None and 0.0 <= x <= 1.0:
        fig.add_vline(x=x, line_dash="dot", line_color=INK, line_width=1.5,
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font_size=12)


# ------------------------------------------------------------------ G1
def _principal_agent(p: dict[str, float]) -> go.Figure:
    w, pr, F, g = p["w"], p["p"], p["F"], p["g"]
    x = np.linspace(0, 1, 101)
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.full_like(x, w), name="Comply: w", line=dict(color=GREEN, width=3),
                    hovertemplate="comply %{y:.2f}<extra></extra>")
    fig.add_scatter(x=x, y=g - F * x, name="Deviate: g - p·F", line=dict(color=RED, width=3),
                    hovertemplate="deviate %{y:.2f}<extra></extra>")
    # where does complying start to pay?
    if F > 0:
        cross = (g - w) / F
    else:
        cross = -1.0 if w >= g else 2.0          # flat lines: one side everywhere
    _regions(fig, cross, good_above=True, good_label="complying pays", bad_label="deviating pays")
    _threshold_line(fig, cross if F > 0 else None, f"p* = {cross:.2f}")
    fig.add_scatter(x=[pr, pr], y=[w, g - F * pr], mode="markers+lines", name="This policy",
                    line=dict(color=INK, dash="dash", width=1), marker=dict(size=11, color=INK),
                    hovertemplate="this policy<extra></extra>")
    fig.add_annotation(x=pr, y=max(w, g - F * pr), text=f"m = {G.margin('principal_agent', p):+.2f}",
                       showarrow=True, arrowhead=0, ax=38, ay=-30, font=dict(size=12, color=INK))
    _layout(fig, "Comply or deviate? Payoffs as detection rises",
            "p: probability deviation is detected", "agent's payoff")
    fig.update_yaxes(range=[min(0, g - F) - 0.1, max(1, g, w) + 0.15])
    return fig


# ------------------------------------------------------------------ G2
def _collective_action(p: dict[str, float]) -> go.Figure:
    t, g, k, x0 = p["t"], p["g"], p["k"], p["x0"]
    x = np.linspace(0, 1, 101)
    y = t - g + k * x
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.maximum(y, 0), fill="tozeroy", fillcolor="rgba(27,127,91,0.25)",
                    line=dict(width=0), name="cooperating pays", hoverinfo="skip")
    fig.add_scatter(x=x, y=np.minimum(y, 0), fill="tozeroy", fillcolor="rgba(178,58,58,0.25)",
                    line=dict(width=0), name="defecting pays", hoverinfo="skip")
    fig.add_scatter(x=x, y=y, name="Gain from cooperating: t - g + k·x",
                    line=dict(color=INK, width=3), hovertemplate="gain %{y:+.2f}<extra></extra>")
    lo, hi = t - g, t - g + k
    xs = (g - t) / k if k > 0 else None
    if lo < 0 < hi and xs is not None:
        _threshold_line(fig, xs, f"tipping point x* = {xs:.0%}")
        kind = "Stag hunt (coordination)"
    elif hi <= 0:
        kind = "Prisoner's dilemma: defection dominates"
    else:
        kind = "Cooperation dominates"
    y0 = t - g + k * x0
    fig.add_scatter(x=[x0], y=[y0], mode="markers", name="Start (x₀)",
                    marker=dict(size=13, color=INK, symbol="circle"),
                    hovertemplate="start x0 = %{x:.0%}, gain %{y:+.2f}<extra></extra>")
    # drift: if cooperating pays at x0, cooperation spreads (x rises); else it unravels
    if abs(y0) > 1e-9:
        target = 1.0 if y0 > 0 else 0.0
        end = x0 + (target - x0) * 0.8
        if abs(end - x0) > 0.03:
            fig.add_annotation(x=end, y=y0, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
                               arrowcolor=GREEN if y0 > 0 else RED, text="")
            fig.add_annotation(x=(x0 + end) / 2, y=y0, yshift=14 if y0 > 0 else -14, showarrow=False,
                               text="cooperation spreads" if y0 > 0 else "cooperation unravels",
                               font=dict(size=11, color=GREEN if y0 > 0 else RED))
    _layout(fig, f"Cooperate or defect? {kind}",
            "x: share of others cooperating", "gain from cooperating vs defecting")
    pad = 0.15
    fig.update_yaxes(range=[min(lo, hi, 0) - pad, max(lo, hi, 0) + pad])
    return fig


# ------------------------------------------------------------------ G3
def _commitment(p: dict[str, float]) -> go.Figure:
    kap, b, c = p["kappa"], p["b"], p["c"]
    x = np.linspace(0, 1, 101)
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.zeros_like(x), name="Wait: 0", line=dict(color=GREY, width=3),
                    hovertemplate="wait 0<extra></extra>")
    fig.add_scatter(x=x, y=x * b - c, name="Respond: κ·b - c", line=dict(color=GREEN, width=3),
                    hovertemplate="respond %{y:+.2f}<extra></extra>")
    cross = c / b if b > 0 else 2.0
    _regions(fig, cross, good_above=True, good_label="responding pays", bad_label="waiting pays")
    _threshold_line(fig, cross if b > 0 else None, f"κ* = {cross:.2f}")
    fig.add_scatter(x=[kap, kap], y=[0, kap * b - c], mode="markers+lines", name="This policy",
                    line=dict(color=INK, dash="dash", width=1), marker=dict(size=11, color=INK),
                    hovertemplate="this policy<extra></extra>")
    fig.add_annotation(x=kap, y=max(0, kap * b - c), text=f"m = {G.margin('commitment', p):+.2f}",
                       showarrow=True, arrowhead=0, ax=38, ay=-30, font=dict(size=12, color=INK))
    if cross > 1:
        fig.add_annotation(x=0.5, y=(b - c) / 2, showarrow=False, font=dict(size=11, color=RED),
                           text="cost exceeds benefit: no credibility level is enough")
    _layout(fig, "Respond or wait? Payoffs as credibility rises",
            "κ: credibility of the commitment", "public's payoff")
    fig.update_yaxes(range=[min(-c, 0) - 0.1, max(b - c, 0) + 0.2])
    return fig


_PAYOFF = {"principal_agent": _principal_agent, "collective_action": _collective_action,
           "commitment": _commitment}


def payoff_chart(game: str, params: dict | None) -> go.Figure | None:
    p = G.clean_params(game, params)
    return _PAYOFF[game](p) if p else None


# ------------------------------------------------------------------ levers
def lever_chart(game: str, params: dict | None) -> go.Figure | None:
    """How much m rises if each parameter moves to its favourable extreme."""
    p = G.clean_params(game, params)
    if not p:
        return None
    gm = G.GAMES[game]
    gains = sorted(G.gains(game, p).items(), key=lambda kv: kv[1])
    bind = G.binding_term(game, p)
    labels = [f"<b>{s}</b>  {_wrap(gm.param(s).name)}" for s, _ in gains]
    top = max(v for _, v in gains)
    fig = go.Figure(go.Bar(
        x=[v for _, v in gains], y=labels, orientation="h",
        marker_color=[GREEN if s == bind else "#a9c9bb" for s, _ in gains],
        text=[f"+{v:.2f}" for _, v in gains], textposition="outside",
        hovertext=[gm.param(s).lever for s, _ in gains], hoverinfo="text"))
    fig.update_layout(
        title=dict(text="Which lever raises the margin most?", font=dict(size=15)),
        height=120 + 56 * len(gains), margin=dict(t=50, b=45, l=10, r=10), plot_bgcolor="white",
        xaxis=dict(title="rise in m if fixed", range=[0, max(0.2, top * 1.35)],
                   showgrid=True, gridcolor="#eee"),
        yaxis=dict(automargin=True))
    return fig


def _wrap(text: str, width: int = 18) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


def params_chart(game: str, params: dict | None, reasons: dict[str, str] | None = None) -> go.Figure | None:
    p = G.clean_params(game, params)
    if not p:
        return None
    gm = G.GAMES[game]
    fig = go.Figure(go.Bar(
        x=[p[q.sym] for q in gm.params], y=[f"{q.sym}: {q.name}" for q in gm.params], orientation="h",
        marker_color=[GREEN if q.favourable else RED for q in gm.params],
        hovertext=[(reasons or {}).get(q.sym, "") for q in gm.params], hoverinfo="text"))
    fig.update_layout(height=60 + 40 * len(gm.params), margin=dict(t=10, b=10, l=10, r=10),
                      xaxis=dict(range=[0, 1]), yaxis=dict(autorange="reversed"), plot_bgcolor="white")
    return fig


# ------------------------------------------------------------------ evidence base
def margin_outcome_chart(df: pd.DataFrame, highlight_margin: float | None = None) -> go.Figure:
    """Every scored record: incentive margin (from design) vs its real outcome."""
    d = df[df["has_outcome"] & df["game_scored"].astype(bool)].copy()
    rng = np.random.default_rng(0)
    d["y"] = d["outcome_direction"].map({o: i for i, o in enumerate(OUTCOME_ORDER)}).astype(float)
    d["yj"] = d["y"] + rng.uniform(-0.18, 0.18, len(d))
    symbols = {"principal_agent": "circle", "collective_action": "square", "commitment": "diamond"}
    fig = go.Figure()
    fig.add_vrect(x0=-1, x1=0, fillcolor=RED_BG, line_width=0, layer="below")
    fig.add_vrect(x0=0, x1=1, fillcolor=GREEN_BG, line_width=0, layer="below")
    for gk, sub in d.groupby("game"):
        fig.add_scatter(
            x=sub["margin"], y=sub["yj"], mode="markers", name=G.GAMES[gk].name,
            marker=dict(size=10, symbol=symbols.get(gk, "circle"), opacity=0.8,
                        color=[OUTCOME_COLOR[o] for o in sub["outcome_direction"]],
                        line=dict(width=1, color="white")),
            text=sub["policy_name"], customdata=sub["outcome_direction"].map(OUTCOME_LABEL),
            hovertemplate="%{text}<br>m = %{x:+.2f} · %{customdata}<extra></extra>",
            showlegend=False)
        # legend entry for the shape only, in a neutral colour (colour = outcome)
        fig.add_scatter(x=[None], y=[None], mode="markers", name=G.GAMES[gk].name,
                        marker=dict(size=10, symbol=symbols.get(gk, "circle"), color=GREY))
    means =d.groupby("outcome_direction")["margin"].mean()
    fig.add_scatter(x=[means.get(o) for o in OUTCOME_ORDER], y=list(range(len(OUTCOME_ORDER))),
                    mode="markers+lines", name="mean margin", line=dict(color=INK, width=1, dash="dot"),
                    marker=dict(size=14, symbol="line-ns-open", color=INK, line=dict(width=3)),
                    hovertemplate="mean m = %{x:+.2f}<extra></extra>")
    if highlight_margin is not None:
        fig.add_vline(x=highlight_margin, line_color=INK, line_dash="dash",
                      annotation_text="this policy", annotation_position="top")
    fig.update_layout(
        title=dict(text="Incentive margin (scored from design only) vs the real outcome",
                   font=dict(size=15)),
        height=380, margin=dict(t=55, b=50, l=10, r=20), plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(title="incentive margin m  (left: incentives work against the policy, right: for it)",
                   range=[-1.05, 1.05], zeroline=True, zerolinecolor="#999", showgrid=True, gridcolor="#eee"),
        yaxis=dict(tickvals=list(range(len(OUTCOME_ORDER))), ticktext=[OUTCOME_LABEL[o] for o in OUTCOME_ORDER],
                   autorange="reversed", showgrid=False))
    return fig
