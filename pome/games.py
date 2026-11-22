"""Three game-theoretic models of how a policy's key players respond.

Every policy is assigned to exactly one game, and the game's parameters are
scored on a 0-1 scale from the policy's *design* (never its outcome). Each
game reduces to one inequality, the **incentive margin** m:

    G1 Principal-agent    m = w + p*F - g        agent complies iff m >= 0
    G2 Collective action  m = t - g + k*x0       user cooperates iff m >= 0
    G3 Commitment         m = kappa*b - c        public responds iff m >= 0

m > 0 means the incentives of the people the policy depends on point toward
it working; m < 0 means the design asks them to act against their own payoff.

The **binding constraint** is the parameter whose move to its favourable
extreme (0 for a cost, 1 for a benefit) would raise m the most, holding the
others fixed. It names the design lever with the largest payoff.

Scale anchors used for every parameter: 0 = none, 0.25 = weak, 0.5 = moderate,
0.75 = strong, 1 = overwhelming. Values are relative, not money amounts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ANCHORS = "0 = none, 0.25 = weak, 0.5 = moderate, 0.75 = strong, 1 = overwhelming"


@dataclass(frozen=True)
class Param:
    sym: str
    name: str
    favourable: int          # 1: higher is better for the policy, 0: lower is better
    guide: str               # what to look for in the design when scoring
    lever: str               # design lever if this is the binding constraint


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    players: str
    question: str            # the strategic question in one line
    formula: str             # margin formula, plain text
    params: list[Param]
    margin_fn: Callable[[dict[str, float]], float]
    payoff_table: str        # markdown 2x2 table for teaching
    threshold_note: str      # the critical value implied by the inequality
    theory_ref: str
    absorbs: list[str] = field(default_factory=list)   # old lenses mapped here

    @property
    def symbols(self) -> list[str]:
        return [p.sym for p in self.params]

    def param(self, sym: str) -> Param:
        return next(p for p in self.params if p.sym == sym)


GAMES: dict[str, Game] = {
    "principal_agent": Game(
        key="principal_agent",
        name="Principal-agent",
        players="Government (principal) -> the agent whose hidden action decides the outcome: "
                "an implementing official, intermediary, firm, taxpayer or beneficiary",
        question="Does the agent do what the policy intends, or divert, shirk or evade "
                 "when the government can't fully observe them?",
        formula="m = w + p*F - g",
        params=[
            Param("w", "reward for complying", 1,
                  "What the agent gains by doing as intended: a conditional payment, wage, "
                  "bonus, or the benefit itself if it is only received by complying.",
                  "Raise or better target the reward for complying: conditional transfers, "
                  "performance pay, bonuses tied to verified delivery."),
            Param("p", "probability deviation is detected", 1,
                  "How visible the agent's action is: digital trails, DBT, biometrics, "
                  "metering, audits, third-party verification. Paper reporting by the "
                  "agent itself is low.",
                  "Raise detection: random and social audits, DBT and e-payment trails, "
                  "independent verification of reported outcomes."),
            Param("F", "penalty if caught", 1,
                  "What happens to an agent caught deviating: fines, prosecution, loss of "
                  "benefit or licence. Low if sanctions are rare, small or politically blocked.",
                  "Make sanctions credible and proportionate: automatic penalties, loss of "
                  "eligibility, publicised enforcement."),
            Param("g", "gain from deviating", 0,
                  "What the agent gains by diverting, shirking, evading or gaming: size of "
                  "rents in the payment chain, discretion at the last mile, value of evasion.",
                  "Cut the gain from deviating: shorten the payment chain, remove last-mile "
                  "discretion, rules-based eligibility, smaller cash pools at each layer."),
        ],
        margin_fn=lambda v: v["w"] + v["p"] * v["F"] - v["g"],
        payoff_table=(
            "| Agent's payoff | Detected (prob. p) | Not detected (1 - p) |\n"
            "|---|---|---|\n"
            "| **Comply** | w | w |\n"
            "| **Deviate** | g - F | g |\n\n"
            "Expected payoff of deviating = g - p*F. The agent complies iff "
            "w >= g - p*F, i.e. **w + p*F - g >= 0** (the incentive-compatibility constraint)."
        ),
        threshold_note="Detection needed for compliance: p* = (g - w) / F.",
        theory_ref="Jensen & Meckling (1976); Becker (1968); Allingham & Sandmo (1972)",
        absorbs=["principal_agent", "tax_compliance", "conditional_incentives", "mechanism_design"],
    ),
    "collective_action": Game(
        key="collective_action",
        name="Collective action",
        players="Many users of a shared resource, network or standard, each deciding "
                "whether to cooperate (conserve, adopt, comply) or defect",
        question="Is it individually rational to cooperate, given what others are doing?",
        formula="m = t - g + k*x0",
        params=[
            Param("t", "policy correction", 1,
                  "How strongly the policy itself rewards cooperating or penalises defecting: "
                  "a Pigouvian tax or fee, enforced quota or ban, metering, or a subsidy for "
                  "cooperating. Zero if the policy only exhorts.",
                  "Price the externality: a tax or fee on defecting, tradable quotas, metering, "
                  "or a payment for cooperating."),
            Param("g", "private gain from defecting", 0,
                  "How much a single user gains by over-using, polluting or not adopting, "
                  "ignoring the cost to others. High when the resource is free at the margin.",
                  "Lower the private gain from defecting: cheap substitutes, alternative "
                  "livelihoods, restricting access to the resource."),
            Param("k", "benefit from others cooperating", 1,
                  "How much cooperating pays off *more* when others also cooperate (network "
                  "effects, shared infrastructure, social norms). 0 for a pure commons.",
                  "Strengthen complementarities: shared infrastructure, interoperability, "
                  "making cooperation visible so norms can form."),
            Param("x0", "expected starting share of others cooperating", 1,
                  "How many users are likely to cooperate at launch: existing adoption, "
                  "mandated early adopters, community institutions already in place.",
                  "Seed a critical mass: start with anchor users or a mandate for early "
                  "adopters so the tipping point is crossed."),
        ],
        margin_fn=lambda v: v["t"] - v["g"] + v["k"] * v["x0"],
        payoff_table=(
            "| Gain from cooperating vs defecting | Others cooperate (x = 1) | Others defect (x = 0) |\n"
            "|---|---|---|\n"
            "| **u(Cooperate) - u(Defect)** | t - g + k | t - g |\n\n"
            "With others cooperating at share x, a user cooperates iff **t - g + k*x >= 0**.\n"
            "* t - g + k < 0: defection dominates (prisoner's dilemma, tragedy of the commons).\n"
            "* t - g < 0 < t - g + k: coordination game (stag hunt) with tipping point "
            "x* = (g - t) / k.\n"
            "* t - g >= 0: cooperation dominates; the policy has closed the gap."
        ),
        threshold_note="Tipping point: x* = (g - t) / k (only when k > 0).",
        theory_ref="Hardin (1968); Olson (1965); Ostrom (1990); Schelling (1960)",
        absorbs=["commons_externality", "coordination"],
    ),
    "commitment": Game(
        key="commitment",
        name="Commitment",
        players="Government -> the public, a market or an organised group that must incur "
                "a cost now in response to a promise or rule that pays off later",
        question="Will people act on the policy, given that the government might reverse it "
                 "or back down?",
        formula="m = kappa*b - c",
        params=[
            Param("kappa", "credibility of the government's commitment", 1,
                  "Belief that the policy will last and be enforced: legislated vs announced, "
                  "track record, independent institutions, cost to the government of reversing, "
                  "strength of organised opposition that could force a climb-down.",
                  "Raise credibility: legislate, delegate to an independent body, phase in "
                  "with pre-announced dates, build broad coalitions before launch."),
            Param("b", "benefit to the public of responding", 1,
                  "What responders gain if the policy holds: lower prices, market access, "
                  "returns on investment, avoided penalties.",
                  "Increase the payoff to responding: front-load benefits, guarantee "
                  "minimum returns, bundle with support services."),
            Param("c", "cost of responding or adjusting now", 0,
                  "Upfront cost to responders: investment, switching, lost rents, disruption, "
                  "political cost for groups asked to give something up.",
                  "Lower the adjustment cost: transition support, compensation for losers, "
                  "gradual phase-in."),
        ],
        margin_fn=lambda v: v["kappa"] * v["b"] - v["c"],
        payoff_table=(
            "| Public's payoff | Government keeps the policy (prob. kappa) | Government reverses (1 - kappa) |\n"
            "|---|---|---|\n"
            "| **Respond** | b - c | -c |\n"
            "| **Wait** | 0 | 0 |\n\n"
            "Expected payoff of responding = kappa*b - c. The public responds iff "
            "**kappa*b - c >= 0**."
        ),
        threshold_note="Credibility needed for a response: kappa* = c / b.",
        theory_ref="Kydland & Prescott (1977); Barro & Gordon (1983); Rubinstein (1982); Schelling (1960)",
        absorbs=["credibility_signalling", "bargaining"],
    ),
}

GAME_KEYS = list(GAMES)
# Old 8-lens tag -> game. Used only as a fallback before scoring and as a sanity check.
OLD_LENS_TO_GAME = {old: g.key for g in GAMES.values() for old in g.absorbs}


# ------------------------------------------------------------------ maths
def clean_params(game: str, params: dict | None) -> dict[str, float] | None:
    """Return the game's parameters clipped to [0, 1], or None if any is missing."""
    if not params:
        return None
    out = {}
    for s in GAMES[game].symbols:
        v = params.get(s)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v != v:   # NaN
            return None
        out[s] = min(1.0, max(0.0, v))
    return out


def margin(game: str, params: dict | None) -> float | None:
    p = clean_params(game, params)
    if p is None:
        return None
    return max(-1.0, min(1.0, GAMES[game].margin_fn(p)))


def gains(game: str, params: dict) -> dict[str, float]:
    """How much m would rise if each parameter moved to its favourable extreme."""
    g = GAMES[game]
    p = clean_params(game, params)
    base = g.margin_fn(p)
    out = {}
    for prm in g.params:
        q = dict(p)
        q[prm.sym] = float(prm.favourable)
        out[prm.sym] = g.margin_fn(q) - base
    return out


def binding_term(game: str, params: dict | None) -> str | None:
    p = clean_params(game, params)
    if p is None:
        return None
    gn = gains(game, p)
    return max(gn, key=gn.get)


def regime(game: str, params: dict | None) -> str:
    """Plain-language reading of the inequality for this parameter set."""
    p = clean_params(game, params)
    if p is None:
        return ""
    p = {s: round(v, 2) for s, v in p.items()}   # same precision as displayed
    if game == "principal_agent":
        dev = p["g"] - p["p"] * p["F"]
        if p["w"] >= dev:
            return "Incentive-compatible: complying pays at least as much as deviating."
        if p["F"] > 0:
            need = (p["g"] - p["w"]) / p["F"]
            if need <= 1:
                return (f"Not incentive-compatible: deviating pays more. Detection would "
                        f"have to rise from {p['p']:.2f} to about {need:.2f}.")
        return "Not incentive-compatible, and no feasible detection rate fixes it without a larger penalty or reward."
    if game == "collective_action":
        lo, hi = p["t"] - p["g"], p["t"] - p["g"] + p["k"]
        if lo >= 0:
            return "Cooperation dominates: the policy has closed the private-social gap."
        if hi < 0:
            return "Defection dominates (prisoner's dilemma): even if everyone else cooperates, each user gains by defecting."
        xs = (p["g"] - p["t"]) / p["k"]
        side = "above" if p["x0"] >= xs else "below"
        return (f"Coordination game (stag hunt): cooperating pays once more than {xs:.0%} of others "
                f"cooperate. The expected starting share ({p['x0']:.0%}) is {side} that tipping point.")
    if game == "commitment":
        if p["b"] <= 0:
            return "No benefit to responding, so no credibility level induces a response."
        need = p["c"] / p["b"]
        if p["kappa"] >= need:
            return f"Credible enough: responding pays when credibility is at least {min(need, 1):.2f}; it is {p['kappa']:.2f}."
        if need > 1:
            return "Even a fully credible commitment wouldn't make responding worthwhile: the cost exceeds the benefit."
        return f"Not credible enough: responding needs credibility of at least {need:.2f}, but it is {p['kappa']:.2f}."
    return ""


def equation(game: str, params: dict | None) -> str:
    """The margin formula with the numbers substituted."""
    p = clean_params(game, params)
    if p is None:
        return GAMES[game].formula
    m = margin(game, p)
    if game == "principal_agent":
        s = f"{p['w']:.2f} + {p['p']:.2f} x {p['F']:.2f} - {p['g']:.2f}"
    elif game == "collective_action":
        s = f"{p['t']:.2f} - {p['g']:.2f} + {p['k']:.2f} x {p['x0']:.2f}"
    else:
        s = f"{p['kappa']:.2f} x {p['b']:.2f} - {p['c']:.2f}"
    return f"m = {s} = {m:+.2f}"


def describe(game: str, params: dict | None, rationale: str = "") -> str:
    """Compact text block for LLM prompts and logs."""
    g = GAMES[game]
    lines = [f"GAME: {g.name}. {g.question}", f"Condition for success: {g.formula} >= 0"]
    p = clean_params(game, params)
    if p:
        for prm in g.params:
            lines.append(f"  {prm.sym} ({prm.name}) = {p[prm.sym]:.2f}")
        b = binding_term(game, p)
        lines += [equation(game, p), regime(game, p),
                  f"Binding constraint: {b} ({g.param(b).name}). Lever: {g.param(b).lever}"]
    if rationale:
        lines.append(f"Why this game: {rationale}")
    return "\n".join(lines)


def scoring_guide() -> str:
    """The three games, formulas and anchors, as given to the scorer (LLM or human)."""
    out = [
        "CHOOSING THE GAME. Almost every policy has an official who implements it, so that "
        "alone does NOT make it principal-agent. Ask whose decision will most decide the "
        "outcome, and apply these rules IN ORDER:\n"
        "1. collective_action if the outcome depends on many users' choices whose effects "
        "spill over onto each other: use of a shared resource (groundwater, air, roads, the "
        "grid, forests, fisheries), pollution, congestion, or adoption where each user gains "
        "more when others also adopt (networks, markets, sanitation norms, standards).\n"
        "2. commitment if the outcome depends on people, firms or organised groups bearing "
        "a cost now based on whether they believe the government will sustain the policy "
        "and not back down: structural or market reforms, long-horizon schemes, bans or "
        "price reforms facing organised opposition, announcements that must be believed.\n"
        "3. principal_agent otherwise: the outcome turns on whether an implementer, "
        "intermediary, taxpayer, firm or beneficiary acts as intended when the government "
        "can't fully observe them (leakage, targeting, evasion, conditional transfers).",
        f"\nSCORING. Score every parameter on a 0-1 scale ({ANCHORS}). Values are relative, not "
        "money amounts. Score from the policy DESIGN only, as it would have looked at launch.",
    ]
    for g in GAMES.values():
        out.append(f"\n### {g.key}: {g.name}\nPlayers: {g.players}\nQuestion: {g.question}\n"
                   f"Condition for success: {g.formula} >= 0")
        for prm in g.params:
            out.append(f"- {prm.sym} = {prm.name}. {prm.guide}")
    return "\n".join(out)
