"""Common locations of the repository."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "input"
RES = ROOT / "results"
LTX = RES / "latex"
FIG = ROOT / "figures"
NPROC = int(os.environ.get("NPROC", "8"))
sys.path.insert(0, str(ROOT / "rqnn"))
