"""Central settings. Change values here instead of hard-coding them elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Elo
ELO_START = 1500.0
ELO_NEW_TEAM = 1450.0      # promoted / unseen teams
ELO_K = 20.0
ELO_HOME_ADV = 65.0
ELO_SEASON_REGRESS = 0.75  # share of rating kept at the start of a new season

# Features
FORM_WINDOWS = (5, 10)
STYLE_CLUSTERS = 4

# Optional per-match stat pairs: (name, home column, away column).
# Only pairs whose columns exist in the data are used. Add your own
# (for example xG or PPDA) by adding columns to the CSVs and listing them here.
STAT_PAIRS = [
    ("shots", "HS", "AS"),
    ("sot", "HST", "AST"),
    ("corners", "HC", "AC"),
    ("fouls", "HF", "AF"),
    ("xg", "HxG", "AxG"),
    ("poss", "HPoss", "APoss"),
    ("ppda", "HPPDA", "APPDA"),
]
STYLE_BASE = ["shots_for_10", "shots_ag_10", "poss_for_10", "ppda_for_10",
              "corners_for_10", "fouls_for_10", "xg_for_10"]

# Training
BURN_IN_SEASONS = 1        # first seasons are used only to warm up Elo and form
TEST_SEASONS = 2           # most recent seasons held out for testing
RANDOM_SEED = 42

# Simulation
N_SIMS = 10_000

# Label mapping
LABELS = {"H": 0, "D": 1, "A": 2}
