# config.py — Project-wide path configuration
# This file sits at the project root and provides paths
# to all other files. No hardcoded usernames anywhere.

from pathlib import Path

# Project root is wherever this file lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Key directories
DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Key files
LEADS_CSV   = DATA_DIR / "Leads.csv"
MODEL_PKL   = MODELS_DIR / "lead_scoring_model.pkl"
SCALER_PKL  = MODELS_DIR / "scaler.pkl"
COLS_PKL    = MODELS_DIR / "model_columns.pkl"

if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data         : {DATA_DIR}")
    print(f"Models       : {MODELS_DIR}")
    print(f"Outputs      : {OUTPUTS_DIR}")
    print(f"Leads CSV    : {LEADS_CSV}")