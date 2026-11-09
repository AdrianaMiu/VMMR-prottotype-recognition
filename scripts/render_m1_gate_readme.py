"""Render reports/m1_gate/README.md from the gate result files (issue M1-16).

Usage: uv run python scripts/render_m1_gate_readme.py
Reads every reports/m1_gate/<track>_<variant>.json and rewrites README.md next to them.
"""

from avr.config import CONFIG_DIR
from avr.pipeline.gate_report import render_from_files

GATE_DIR = CONFIG_DIR.parent / "reports" / "m1_gate"


def main() -> None:
    paths = sorted(GATE_DIR.glob("track_*_p*.json"))
    readme = GATE_DIR / "README.md"
    readme.write_text(render_from_files(paths))
    print(f"wrote {readme} from {len(paths)} result file(s)")


if __name__ == "__main__":
    main()
