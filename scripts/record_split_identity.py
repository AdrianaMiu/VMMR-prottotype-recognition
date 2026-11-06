"""Write data/manifests/track_a_split_identity.json from the local CompCars release.

The dataset root comes from configs/datasets (COMPCARS_ROOT overrides it).
Usage: uv run python scripts/record_split_identity.py
"""

from datetime import date

from avr.config import CONFIG_DIR, load_config
from avr.data.split_identity import record_split_identity

REPO_ROOT = CONFIG_DIR.parent
OUTPUT = REPO_ROOT / "data" / "manifests" / "track_a_split_identity.json"


def main() -> None:
    root = (REPO_ROOT / load_config().dataset.root).resolve()
    record = record_split_identity(root, recorded_at=date.today())
    OUTPUT.write_text(record.model_dump_json(indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(record.files)} files) from {root}")


if __name__ == "__main__":
    main()
