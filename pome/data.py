"""Loading, cleaning and feature derivation for policy_records.csv."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from . import config as C
from . import games as G

# Rows that describe the same underlying policy under a different name.
# Siblings must never be used as precedents for each other in the backtest,
# otherwise the system "predicts" a policy by looking up its own outcome.
FAMILY_OVERRIDES = {
    "farm laws 2020 repeal": "farm laws 2020",
}

TEXT_COLS = [
    "record_id", "policy_name", "sector", "sub_sector", "level", "state",
    "mechanism_type", "target_group", "description", "stated_goal",
    "outcome_summary", "outcome_direction", "outcome_metric", "outcome_value",
    "side_effects", "public_reaction", "evidence_type", "evidence_quality",
    "game_theory_lens", "game_type", "source_title", "source_url", "verified",
]


def policy_family(name: str) -> str:
    s = name.lower()
    s = re.sub(r"\(.*?\)", " ", s)      # drop parentheticals
    s = s.split(" - ")[0]               # drop " - fiscal cost" style suffixes
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return FAMILY_OVERRIDES.get(s, s)


def evidence_weight(evidence_type: str, evidence_quality: str, verified: str) -> float:
    return (
        C.EVIDENCE_TYPE_WEIGHT.get(evidence_type, 0.4)
        * C.EVIDENCE_QUALITY_MULT.get(evidence_quality, 0.6)
        * C.VERIFIED_MULT.get(verified, 0.8)
    )


def design_text(r: dict) -> str:
    """Text describing the policy *design* only, never its outcome.

    Stored records and new queries are embedded with the same template, so
    retrieval matches design to design and the outcome can't leak into the
    query side of the backtest."""
    where = r.get("level") or ""
    if r.get("state"):
        where = f"{where} ({r['state']})"
    parts = [
        f"{r.get('policy_name', '')}.",
        f"Sector: {r.get('sector', '')} / {r.get('sub_sector', '')}.",
        f"Mechanism: {str(r.get('mechanism_type', '')).replace('_', ' ')}.",
        f"Level: {where}." if where else "",
        f"Target group: {r.get('target_group', '')}.",
        f"Design: {r.get('description', '')}",
        f"Goal: {r.get('stated_goal', '')}",
    ]
    return " ".join(p for p in parts if p and not p.endswith(": ."))


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_records(path: Path | str = C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in TEXT_COLS:
        if col in df:
            df[col] = df[col].fillna("").astype(str).str.strip()
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce").astype("Int64")
    df["year_end"] = pd.to_numeric(df["year_end"], errors="coerce").astype("Int64")

    df["family"] = df["policy_name"].map(policy_family)
    df["evidence_weight"] = [
        evidence_weight(t, q, v)
        for t, q, v in zip(df.evidence_type, df.evidence_quality, df.verified)
    ]
    df["has_outcome"] = df["outcome_direction"].isin(C.OUTCOME_CLASSES)
    df["outcome_score"] = df["outcome_direction"].map(C.OUTCOME_SCORE)
    df["design_text"] = [design_text(r) for r in df.to_dict("records")]

    if df["record_id"].duplicated().any():
        raise ValueError("record_id must be unique")
    df = attach_games(df)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ game model
PARAM_COLS = ["p1", "p2", "p3", "p4"]


def load_game_params(path: Path | str = C.GAME_PARAMS_PATH) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    gp = pd.read_csv(path, dtype={"record_id": str, "game": str})
    return gp.drop_duplicates("record_id", keep="last")


def row_params(row: dict) -> dict[str, float]:
    """Map the generic p1..p4 columns of game_params.csv to the game's symbols."""
    syms = G.GAMES[row["game"]].symbols
    return {s: row.get(c) for s, c in zip(syms, PARAM_COLS)}


def attach_games(df: pd.DataFrame) -> pd.DataFrame:
    """Add game, game_params, margin, binding_term, game_scored, game_reviewed.

    Margins are always recomputed from the parameters, so hand edits to
    game_params.csv stay consistent. Records not yet scored fall back to the
    old lens tag mapped onto the three games, with no parameters."""
    gp = load_game_params()
    by_id = {} if gp is None else {r["record_id"]: r for r in gp.to_dict("records")}
    games, params, margins, binds, scored, reviewed = [], [], [], [], [], []
    for r in df.to_dict("records"):
        g = by_id.get(r["record_id"])
        if g is not None and g.get("game") in G.GAMES:
            p = G.clean_params(g["game"], row_params(g))
            games.append(g["game"])
            reviewed.append(str(g.get("reviewed", "n")).strip().lower() == "y")
        else:
            p = None
            games.append(G.OLD_LENS_TO_GAME.get(r.get("game_type", ""), "principal_agent"))
            reviewed.append(False)
        params.append(p or {})
        margins.append(G.margin(games[-1], p) if p else float("nan"))
        binds.append(G.binding_term(games[-1], p) if p else "")
        scored.append(p is not None)
    df["game"] = games
    df["game_params"] = params
    df["margin"] = margins
    df["binding_term"] = binds
    df["game_scored"] = scored
    df["game_reviewed"] = reviewed
    return df
