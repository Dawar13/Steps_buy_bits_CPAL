"""Make the repository root importable when a script is run as `python scripts/<name>.py`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def str_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]
