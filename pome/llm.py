"""Optional LLM layer (Groq): game-model scoring, re-ranking, cited prediction, and a
citation-faithfulness judge.

Everything returns None if no GROQ_API_KEY is configured or the
call fails, so the pipeline falls back to the deterministic
reference-class forecaster. The LLM is the "inside view" that adjusts the
outside-view base rate with case-specific reasoning; it never replaces it."""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel

from . import config as C
from . import games as G
from .retrieve import PolicyQuery, Precedent

GameKey = Literal["principal_agent", "collective_action", "commitment"]
Outcome = Literal["improved", "mixed", "no_clear_effect", "worsened"]

SYSTEM = (
    "You are a policy economist supporting Indian government decision makers. "
    "You reason from documented precedents and standard game theory "
    "(principal-agent incentive compatibility, collective action, and commitment / "
    "credibility). You are calibrated: you "
    "state uncertainty honestly, you do not invent facts, and every empirical claim "
    "you make about a past policy cites its record_id from the evidence provided."
)


# ------------------------------------------------------------------ schemas
class ParamScore(BaseModel):
    symbol: str
    value: float          # 0..1
    reason: str           # one line, citing the design


class GameScore(BaseModel):
    game: GameKey
    rationale: str        # who the players are and why this game decides the outcome
    params: list[ParamScore]

    def param_dict(self) -> dict[str, float]:
        return {p.symbol: p.value for p in self.params}

    def reason_dict(self) -> dict[str, str]:
        return {p.symbol: p.reason for p in self.params}


class RerankItem(BaseModel):
    record_id: str
    fit: float            # 0..1, how good an analogue this is
    shared: str
    differs: str


class Rerank(BaseModel):
    items: list[RerankItem]


class CitedClaim(BaseModel):
    claim: str
    record_ids: list[str]


class RiskItem(BaseModel):
    risk: str
    likelihood: Literal["high", "medium", "low"]
    mitigation: str
    record_ids: list[str]


class LLMForecast(BaseModel):
    outcome_direction: Outcome
    p_improved: float
    p_mixed: float
    p_no_clear_effect: float
    p_worsened: float
    adjustment_from_base_rate: str   # why the inside view moves away from the outside view
    expected_effects: list[CitedClaim]
    risks: list[RiskItem]
    design_recommendations: list[str]
    key_uncertainties: list[str]

    def probs(self) -> dict[str, float]:
        raw = {
            "improved": max(self.p_improved, 0.0), "mixed": max(self.p_mixed, 0.0),
            "no_clear_effect": max(self.p_no_clear_effect, 0.0),
            "worsened": max(self.p_worsened, 0.0),
        }
        s = sum(raw.values()) or 1.0
        return {k: v / s for k, v in raw.items()}


class Judgement(BaseModel):
    supported: Literal["yes", "partial", "no"]
    reason: str


# ------------------------------------------------------------------ helpers
def _precedent_block(precs: list[Precedent]) -> str:
    rows = []
    for p in precs:
        r = p.record
        rows.append({
            "record_id": r["record_id"], "policy": r["policy_name"],
            "similarity": round(p.similarity, 3), "year": None if r.get("year_start") is None
            or str(r.get("year_start")) == "<NA>" else int(r["year_start"]),
            "level_state": f"{r['level']} {r['state']}".strip(),
            "mechanism": r["mechanism_type"], "game": r.get("game"),
            "incentive_margin": None if r.get("margin") is None or r.get("margin") != r.get("margin")
            else round(float(r["margin"]), 2),
            "design": r["description"], "goal": r["stated_goal"],
            "outcome_direction": r["outcome_direction"], "outcome": r["outcome_summary"],
            "metric": f"{r['outcome_metric']}: {r['outcome_value']}",
            "side_effects": r["side_effects"], "public_reaction": r["public_reaction"],
            "evidence": f"{r['evidence_type']} ({r['evidence_quality']} quality)",
        })
    return json.dumps(rows, indent=1, ensure_ascii=False)


def _query_block(q: PolicyQuery) -> str:
    return json.dumps({k: v for k, v in q.__dict__.items()
                       if v not in (None, "", {}) and k not in ("game", "game_params")},
                      indent=1, ensure_ascii=False)


class PolicyLLM:
    """LLM layer backed by Groq's chat completions API (free tier).

    Uses JSON mode and validates the reply against the pydantic schema,
    retrying once with the validation error if the model's JSON is off."""

    def __init__(self, model: str | None = None):
        from groq import Groq

        self.model = model or os.environ.get("GROQ_MODEL") or C.LLM_MODEL
        self.client = Groq(max_retries=3)   # retries 429s from free-tier rate limits
        self.calls = 0
        self.errors: list[str] = []

    @staticmethod
    def available() -> bool:
        if not os.environ.get("GROQ_API_KEY"):
            return False
        try:
            import groq  # noqa: F401
        except ImportError:
            return False
        return True

    def _parse(self, schema, prompt: str, effort: str = "medium", max_tokens: int = 4096):
        # gpt-oss models accept reasoning_effort (low/medium/high); others don't.
        import groq
        from pydantic import ValidationError

        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": SYSTEM + "\n\nRespond with a single JSON object "
             "that conforms to this JSON Schema, with no prose or markdown around it:\n"
             + schema_json},
            {"role": "user", "content": prompt},
        ]
        extra = {"reasoning_effort": effort} if "gpt-oss" in self.model else {}
        text = ""
        for attempt in range(2):
            try:
                self.calls += 1
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=max_tokens,
                    **extra,
                )
                text = resp.choices[0].message.content or ""
                return schema.model_validate_json(text)
            except ValidationError as e:
                self.errors.append(f"ValidationError: {e.errors()[:3]}")
                if attempt == 0:
                    messages += [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "That JSON did not match the schema: "
                         f"{e}. Return the corrected JSON object only."},
                    ]
                    continue
            except groq.APIStatusError as e:
                self.errors.append(f"{e.status_code}: {e.message}")
            except groq.APIConnectionError as e:
                self.errors.append(f"connection: {e}")
            except Exception as e:
                self.errors.append(f"{type(e).__name__}: {e}")
            return None
        return None

    # -------------------------------------------------------------- tasks
    def score_game(self, q: PolicyQuery) -> GameScore | None:
        """Assign one of the three games and score its parameters from the design.

        Only design fields are sent (never outcomes), so the scores can be used
        in the backtest without leaking the answer."""
        prompt = (
            "Model the policy below as ONE of three games, the strategic interaction that "
            "will most decide whether it works, then score that game's parameters.\n\n"
            + G.scoring_guide() +
            "\n\nPOLICY (design only)\n" + _query_block(q) +
            "\n\nReturn: game (one key), a two-sentence rationale naming the players and "
            "their incentives, and params: one entry per parameter of the chosen game "
            "(use exactly its symbols), each with value in [0, 1] and a one-line reason "
            "that refers to a feature of the design."
        )
        out = self._parse(GameScore, prompt, effort="medium", max_tokens=4000)
        if out is None:
            return None
        want = set(G.GAMES[out.game].symbols)
        if set(out.param_dict()) != want:
            self.errors.append(f"score_game: got symbols {sorted(out.param_dict())}, want {sorted(want)}")
            return None
        return out

    def rerank(self, q: PolicyQuery, precs: list[Precedent]) -> Rerank | None:
        prompt = (
            "Score each candidate precedent as an analogue for the new policy. "
            "fit=1 means the same mechanism, incentives and context; 0 means only "
            "superficially similar wording. Judge the mechanism and the incentives, "
            "not shared keywords.\n\nNew policy:\n" + _query_block(q) +
            "\n\nCandidates:\n" + _precedent_block(precs) +
            "\n\nReturn one item per candidate with record_id, fit, what is shared, "
            "and what differs."
        )
        return self._parse(Rerank, prompt, effort="low", max_tokens=4096)

    def predict(self, q: PolicyQuery, precs: list[Precedent], game_text: str,
                base: dict[str, float], rc_probs: dict[str, float]) -> LLMForecast | None:
        """game_text: games.describe(...) for the policy (game, parameters, margin)."""
        prompt = f"""Forecast the outcome of a new Indian policy.

NEW POLICY
{_query_block(q)}

GAME MODEL (incentive margin m > 0 means the key players' incentives favour success)
{game_text}

OUTSIDE VIEW (statistical, from the precedents below)
Corpus base rate: {json.dumps({k: round(v, 2) for k, v in base.items()})}
Evidence-weighted reference-class forecast: {json.dumps({k: round(v, 2) for k, v in rc_probs.items()})}

PRECEDENTS (the only empirical evidence you may cite)
{_precedent_block(precs)}

INSTRUCTIONS
1. Start from the reference-class forecast. Move away from it only for
   specific, stated reasons about how this policy's design or context differs,
   and explain the move in adjustment_from_base_rate. Use the game model's binding
   constraint as one input, not as a verdict.
2. Give probabilities for improved / mixed / no_clear_effect / worsened that sum to 1.
3. expected_effects: 3-5 concrete claims, each citing the record_ids that support it.
4. risks: a risk register of 3-6 items, each with likelihood, a mitigation
   (a design lever, ideally one that raises the incentive margin), and supporting record_ids.
5. design_recommendations: 3-5 actionable changes to the policy design.
6. key_uncertainties: what evidence would most change this forecast.
7. Cite only record_ids from the precedent list. Weight RCT and quasi-experimental
   evidence above news and official claims."""
        return self._parse(LLMForecast, prompt, effort="medium", max_tokens=8000)

    def predict_no_retrieval(self, q: PolicyQuery) -> LLMForecast | None:
        """Baseline 1: the same model with no precedents (closed-book)."""
        prompt = (
            "Forecast the outcome of this Indian policy from your own knowledge only. "
            "No precedents are provided, so return empty record_ids lists.\n\n"
            + _query_block(q) +
            "\n\nGive probabilities for improved / mixed / no_clear_effect / worsened."
        )
        return self._parse(LLMForecast, prompt, effort="medium", max_tokens=8000)

    def judge(self, claim: str, recs: list[dict]) -> Judgement | None:
        """Citation faithfulness: does the cited evidence support the claim?"""
        ev = json.dumps([{k: r[k] for k in ("record_id", "policy_name", "outcome_summary",
                                             "outcome_value", "side_effects", "public_reaction")}
                         for r in recs], ensure_ascii=False, indent=1)
        prompt = (
            "Does the cited evidence support the claim? Be strict: 'yes' only if the "
            "evidence states it, 'partial' if it is consistent but goes beyond the "
            "evidence, 'no' if it is unsupported or contradicted.\n\n"
            f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}"
        )
        return self._parse(Judgement, prompt, effort="low", max_tokens=2000)
