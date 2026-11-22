"""Forecast a new policy from the command line and write a decision memo.

Example:
  python scripts/predict.py --name "Free electricity for farm pumps" \
      --description "State provides unmetered free power to agricultural pump sets" \
      --goal "Reduce farmers' irrigation costs" --sector energy_water_environment \
      --mechanism subsidy --level state --state Punjab
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pome import config as C  # noqa: E402
from pome.memo import render_memo  # noqa: E402
from pome.pipeline import PolicyPredictor  # noqa: E402
from pome.retrieve import PolicyQuery  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--description", required=True)
ap.add_argument("--goal", default="")
ap.add_argument("--target", default="")
ap.add_argument("--sector", default="")
ap.add_argument("--mechanism", default="")
ap.add_argument("--level", default="")
ap.add_argument("--state", default="")
ap.add_argument("--game", default=None, choices=C.GAMES)
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--out", default=None, help="write the memo to this .md file")
a = ap.parse_args()

q = PolicyQuery(policy_name=a.name, description=a.description, stated_goal=a.goal,
                target_group=a.target, sector=a.sector, mechanism_type=a.mechanism,
                level=a.level, state=a.state, game=a.game)
pred = PolicyPredictor(use_llm=False if a.no_llm else None)
res = pred.predict(q)
memo = render_memo(res)
if a.out:
    Path(a.out).write_text(memo, encoding="utf-8")
    print(f"Memo written to {a.out}")
else:
    print(memo)
if pred.llm and pred.llm.errors:
    print("\nLLM errors:", pred.llm.errors[:3], file=sys.stderr)
