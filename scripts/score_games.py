"""Draft the game model for every record -> game_params.csv (for human review).

For each record the LLM picks one of the three games (pome/games.py) and
scores its parameters on a 0-1 scale, from the DESIGN fields only (name,
sector, mechanism, level, target group, description, goal). Outcome text is
never shown, so the scores can be used in the backtest without leakage.

Review protocol
  1. Run this script. It writes game_params.csv with reviewed = n.
  2. Open game_params.csv. For each row check the game and each p1..p4 value
     against the legend and reason columns; edit anything you disagree with.
  3. Set reviewed = y on rows you have checked. Re-running this script never
     touches reviewed rows. Margins are recomputed from p1..p4 when the data
     is loaded, so you never need to edit margin by hand.

Usage
  python scripts/score_games.py            # score rows not yet in the file
  python scripts/score_games.py --redo     # re-score every unreviewed row
  python scripts/score_games.py --only pm_kisan_income_support_1
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pome import config as C  # noqa: E402
from pome import games as G  # noqa: E402
from pome.data import PARAM_COLS, load_game_params, load_records  # noqa: E402
from pome.llm import PolicyLLM  # noqa: E402
from pome.retrieve import PolicyQuery  # noqa: E402

COLS = (["record_id", "policy_name", "game"] + PARAM_COLS + ["margin", "binding_term", "legend"]
        + [f"why_{c}" for c in PARAM_COLS] + ["rationale", "old_game_type", "reviewed"])

ap = argparse.ArgumentParser()
ap.add_argument("--redo", action="store_true", help="re-score every unreviewed row")
ap.add_argument("--only", nargs="*", default=None, help="record_ids to (re)score")
a = ap.parse_args()

if not PolicyLLM.available():
    sys.exit("GROQ_API_KEY not set (see .env.example).")
llm = PolicyLLM()
df = load_records()
old = load_game_params()
rows = {} if old is None else {r["record_id"]: r for r in old.to_dict("records")}


def save():
    out = pd.DataFrame([rows[i] for i in df["record_id"] if i in rows]).reindex(columns=COLS)
    out.to_csv(C.GAME_PARAMS_PATH, index=False)


todo = []
for r in df.to_dict("records"):
    rid = r["record_id"]
    prev = rows.get(rid)
    if prev is not None and str(prev.get("reviewed", "n")).strip().lower() == "y":
        continue
    if a.only is not None and rid not in a.only:
        continue
    if prev is not None and not a.redo and a.only is None:
        continue
    todo.append(r)

print(f"Scoring {len(todo)} record(s) with {llm.model} ...")
for n, r in enumerate(todo, 1):
    q = PolicyQuery.from_record(r)          # design fields only
    gs = llm.score_game(q)
    if gs is None:
        print(f"  [{n}/{len(todo)}] {r['record_id']}: FAILED ({llm.errors[-1] if llm.errors else '?'})")
        continue
    p = G.clean_params(gs.game, gs.param_dict())
    syms = G.GAMES[gs.game].symbols
    why = gs.reason_dict()
    row = {
        "record_id": r["record_id"], "policy_name": r["policy_name"], "game": gs.game,
        "margin": round(G.margin(gs.game, p), 3), "binding_term": G.binding_term(gs.game, p),
        "legend": " | ".join(f"{c}={s} ({G.GAMES[gs.game].param(s).name})" for c, s in zip(PARAM_COLS, syms)),
        "rationale": gs.rationale, "old_game_type": r.get("game_type", ""), "reviewed": "n",
    }
    for c in PARAM_COLS:
        row[c] = None
        row[f"why_{c}"] = ""
    for c, s in zip(PARAM_COLS, syms):
        row[c] = round(p[s], 2)
        row[f"why_{c}"] = why.get(s, "")
    rows[r["record_id"]] = row
    save()
    print(f"  [{n}/{len(todo)}] {r['record_id'][:44]:44s} {gs.game:18s} m={row['margin']:+.2f} "
          f"binding={row['binding_term']}")

save()
out = pd.read_csv(C.GAME_PARAMS_PATH)
print(f"\nWrote {len(out)} rows to {C.GAME_PARAMS_PATH.name} "
      f"({(out['reviewed'] == 'y').sum()} reviewed).")
print("\nGames assigned:")
print(out["game"].value_counts().to_string())
agree = (out["old_game_type"].map(G.OLD_LENS_TO_GAME) == out["game"])
print(f"\nAgreement with the old 8-lens tags mapped to 3 games: {agree.mean():.0%} "
      f"({agree.sum()}/{len(out)})")
print(pd.crosstab(out["old_game_type"], out["game"]).to_string())
if llm.errors:
    print(f"\nLLM errors ({len(llm.errors)}): {llm.errors[:3]}")
