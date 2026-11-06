"""M0 global class mean prototypes (PROTOCOL.md §11 "M0 - Global class mean").

``p_c = normalize(mean(z_i for i in class c))`` over individually L2-normalised support
embeddings. Prototypes are returned in lexicographic class-id order so the prototype order
is stable regardless of the order of the support rows (§23 "prototype order is stable").
The guard ``query_image_ids`` enforces §23 "no query embedding enters prototype
construction": any support row whose image id is a query raises.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_global_mean_prototypes(
    embeddings: np.ndarray,
    class_ids: Sequence[str],
    image_ids: Sequence[str],
    query_image_ids: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(prototypes [C, d] float32 L2-normalised, class ids sorted lexicographically)``."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [n, d], got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if len(class_ids) != n or len(image_ids) != n:
        raise ValueError(
            f"{n} embeddings but {len(class_ids)} class ids and {len(image_ids)} image ids"
        )
    if n == 0:
        raise ValueError("at least one support embedding is required")
    if query_image_ids:
        leaked = sorted(set(image_ids) & set(query_image_ids))
        if leaked:
            raise ValueError(
                f"{len(leaked)} query image id(s) found among support rows: {leaked[:5]}"
            )

    class_array = np.asarray(list(class_ids), dtype=object)
    sorted_ids = sorted(set(class_ids))
    prototypes = np.empty((len(sorted_ids), embeddings.shape[1]), dtype=np.float32)
    for row, class_id in enumerate(sorted_ids):
        mean = embeddings[class_array == class_id].mean(axis=0, dtype=np.float64)
        norm = np.linalg.norm(mean)
        if norm == 0:
            raise ValueError(f"class {class_id!r}: support embeddings sum to the zero vector")
        prototypes[row] = (mean / norm).astype(np.float32)
    return prototypes, sorted_ids
