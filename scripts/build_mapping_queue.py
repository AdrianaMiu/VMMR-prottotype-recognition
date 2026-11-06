"""Build the generation-mapping curation queue, reviewer sheet and coverage report (§6).

Reads data/manifests/compcars_web_images.parquet (build it first with
scripts/build_image_manifest.py) and writes
data/mappings/generation_mapping_queue_v<ver>.csv, ..._counts.csv,
data/mappings/reviewer_sheet_v<ver>.csv and reports/mapping/v<ver>/coverage.md.
Usage: uv run python scripts/build_mapping_queue.py
"""

import time

import pandas as pd

from avr.config import CONFIG_DIR
from avr.generation_mapping.coverage import coverage_stats, render_report
from avr.generation_mapping.queue import (
    REVIEWER_SHEET_NAME,
    build_queue,
    build_reviewer_sheet,
    write_queue,
)

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / "compcars_web_images.parquet"
MAPPINGS_DIR = REPO_ROOT / "data" / "mappings"
REPORTS_DIR = REPO_ROOT / "reports" / "mapping"
MAPPING_VERSION = "0.1.0"
FILE_VERSION = "0.1"


def main() -> None:
    started = time.perf_counter()
    manifest = pd.read_parquet(MANIFEST)
    queue = build_queue(manifest)
    out = write_queue(queue, MAPPINGS_DIR, version=FILE_VERSION)
    sheet_path = MAPPINGS_DIR / f"{REVIEWER_SHEET_NAME}_v{FILE_VERSION}.csv"
    build_reviewer_sheet(queue).to_csv(sheet_path, index=False)
    stats = coverage_stats(manifest, queue)
    report_dir = REPORTS_DIR / f"v{FILE_VERSION}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "coverage.md"
    report_path.write_text(render_report(stats, out.queue, version=MAPPING_VERSION))
    elapsed = time.perf_counter() - started
    print(
        f"wrote {out.queue.relative_to(REPO_ROOT)} ({len(queue)} rows, "
        f"{stats.model_count} models), {sheet_path.relative_to(REPO_ROOT)}, "
        f"{report_path.relative_to(REPO_ROOT)} in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
