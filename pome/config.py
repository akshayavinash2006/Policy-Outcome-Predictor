"""Central configuration. Every modelling assumption lives here so it can be
cited, defended and varied in sensitivity analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "policy_records.csv"
GAME_PARAMS_PATH = ROOT / "game_params.csv"
ARTIFACTS = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS / "index"

# ---------------------------------------------------------------- models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"   # Groq; override with GROQ_MODEL env var

# ---------------------------------------------------------------- outcomes
# Ordered from best to worst. "unknown" rows are kept for retrieval context
# but never vote on an outcome and are never backtest targets.
OUTCOME_CLASSES = ["improved", "mixed", "no_clear_effect", "worsened"]
# Net-outcome index used for the expected-value summary (range -1 .. +1).
OUTCOME_SCORE = {"improved": 1.0, "mixed": 0.0, "no_clear_effect": 0.0, "worsened": -1.0}

# Three game-theoretic models (pome/games.py). Each record's game and parameters
# are scored from its design text into GAME_PARAMS_PATH, then human-reviewed.
GAMES = ["principal_agent", "collective_action", "commitment"]

# ---------------------------------------------------------------- evidence hierarchy
# Loosely follows the Maryland Scientific Methods Scale (Sherman et al., 1998):
# designs that identify causal effects more credibly get more weight.
EVIDENCE_TYPE_WEIGHT = {
    "rct": 1.00,
    "quasi_experimental": 0.85,
    "audit": 0.70,          # CAG audits: systematic, but descriptive
    "official_data": 0.60,  # administrative statistics, no counterfactual
    "study": 0.55,          # surveys / before-after studies
    "news": 0.35,
}
EVIDENCE_QUALITY_MULT = {"high": 1.00, "medium": 0.75, "low": 0.50}
VERIFIED_MULT = {"yes": 1.00, "partial": 0.80}

# ---------------------------------------------------------------- retrieval
TOP_K = 8
# Relevance gate: a precedent must be more similar to the query than this
# quantile of all pairwise policy similarities in the corpus. Computed at
# index-build time so it adapts to whichever embedding backend is in use.
THRESHOLD_QUANTILE = 0.80
MIN_PRECEDENTS = 2           # fewer above-threshold matches -> abstain
# Soft metadata boosts added to cosine for ranking only (not for the gate).
# Soft rather than hard filters because the corpus is small (57 rows).
BOOSTS = {"game": 0.06, "mechanism_type": 0.04, "sector": 0.02}
# Extra ranking boost for precedents whose incentive margin is close to the
# query's: MARGIN_BOOST * (1 - |m_query - m_precedent| / 2).
MARGIN_BOOST = 0.04
SIM_TEMPERATURE = 0.05       # sharpness of similarity weighting

# ---------------------------------------------------------------- forecasting
# Bayesian shrinkage: precedents are combined with the base rate as if the
# base rate were worth PRIOR_STRENGTH fully-weighted observations.
PRIOR_STRENGTH = 2.0

# ---------------------------------------------------------------- evaluation
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42
RANDOM_BASELINE_DRAWS = 25
