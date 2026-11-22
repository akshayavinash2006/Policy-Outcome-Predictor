"""Embed the policy records and save the vector index."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pome.index import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="auto", choices=["auto", "sbert", "tfidf"])
args = ap.parse_args()
store = build_index(args.backend)
off = store.pairwise_offdiag()
print(f"Indexed {len(store.df)} records with {store.embedder.label} ({store.vectors.shape[1]} dims)")
print(f"Pairwise similarity: median {sorted(off)[len(off)//2]:.3f}, max {off.max():.3f}")
print(f"Relevance gate (q={0.8}): {store.threshold:.3f}")
