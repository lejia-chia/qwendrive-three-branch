"""Repository and upstream runtime paths, independent of current directory."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Qwen-Drive-1.0"
if not UPSTREAM.is_dir():
    raise RuntimeError(f"missing upstream source: {UPSTREAM}")
sys.path.insert(0, str(UPSTREAM / "src"))
MODEL = ROOT / "model"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
