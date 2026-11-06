"""Write data/manifests/track_a_split_identity.json from the local CompCars release.

Usage: uv run python scripts/record_split_identity.py [COMPCARS_ROOT]
"""

import sys
from datetime import date
from pathlib import Path

from avr.data.split_identity import record_split_identity

root = Path(sys.argv[1] if len(sys.argv) > 1 else "../compcar")
out = Path("data/manifests/track_a_split_identity.json")
record = record_split_identity(root, recorded_at=date.today().isoformat())
out.write_text(record.model_dump_json(indent=2) + "\n")
print(f"wrote {out} ({len(record.files)} files)")
