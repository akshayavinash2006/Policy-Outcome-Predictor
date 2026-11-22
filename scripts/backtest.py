"""Run the leakage-controlled backtest and save metrics to artifacts/backtest/."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pome.evaluate import BacktestConfig, coverage_curve, run_backtest, save  # noqa: E402
from pome.index import load_or_build_index  # noqa: E402
from pome.llm import PolicyLLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="both", choices=["cutoff", "loo", "both"])
ap.add_argument("--llm", action="store_true", help="also run the LLM (RAG and closed-book); costs API calls")
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()

store = load_or_build_index()
llm = PolicyLLM() if a.llm else None
for mode in (["cutoff", "loo"] if a.mode == "both" else [a.mode]):
    print(f"\n=== Backtest: {mode} ({store.embedder.label}) ===")
    res, m = run_backtest(store, BacktestConfig(mode=mode, k=a.k, use_llm=a.llm), llm)
    curve = coverage_curve(store, mode)
    save(res, m, mode, curve)
    print(f"\nTargets: {m['n_targets']}  game acc (kNN): {m['game_accuracy_knn']:.2f}  "
          f"retrieval hit rate: {m['retrieval_hit_rate']:.2f}")
    print(f"{'method':16s} {'acc':>6s} {'95% CI':>14s} {'cover':>6s} {'acc|ans':>8s} "
          f"{'F1':>5s} {'Brier':>6s} {'BSS':>6s} {'logloss':>7s}")
    for name, s in m["methods"].items():
        lo, hi = s["accuracy_ci95"]
        aa = s["accuracy_answered"]
        print(f"{name:16s} {s['accuracy']:6.2f} [{lo:4.2f}, {hi:4.2f}] {s['coverage']:6.2f} "
              f"{(aa if aa is not None else float('nan')):8.2f} {s['macro_f1']:5.2f} "
              f"{s['brier']:6.3f} {s['brier_skill_vs_climatology']:+6.2f} {s['log_loss']:7.3f}")
    for k in ("game_accuracy_llm", "citation_validity", "citation_faithfulness_yes",
              "citation_faithfulness_yes_or_partial"):
        if k in m:
            print(f"{k}: {m[k]:.2f}")
    print("\nCoverage vs accuracy (relevance-gate sweep):")
    print(curve.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
if llm:
    print(f"\nLLM calls: {llm.calls}, errors: {len(llm.errors)} {llm.errors[:3]}")
