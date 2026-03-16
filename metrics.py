"""
metrics.py — Metric calculations for the Leaf Adjuvant Analyser.

All functions are pure (no GUI, no I/O) and operate on numpy arrays.
"""

import numpy as np


def coverage_percent(contacted_mask: np.ndarray, leaf_mask: np.ndarray) -> float:
    """
    Percentage of the leaf area that shows spray contact.

    contacted_mask: binary uint8 (255 = contacted)
    leaf_mask:      binary uint8 (255 = leaf)
    """
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    if leaf_pixels == 0:
        return 0.0
    contacted_pixels = int(np.count_nonzero(contacted_mask))
    return min((contacted_pixels / leaf_pixels) * 100.0, 100.0)


def uniformity_score(contacted_mask: np.ndarray, leaf_mask: np.ndarray,
                     grid_n: int = 8) -> float:
    """
    Shannon-entropy-based uniformity of spray distribution across a spatial grid.

    The leaf is divided into grid_n × grid_n cells.  For each cell that
    contains leaf pixels the coverage fraction (contacted / leaf) is computed.
    The normalised Shannon entropy of these per-cell fractions is returned
    on a 0–1 scale (0 = all coverage concentrated in one cell, 1 = perfectly
    uniform across every cell).

    Returns 0.0 if there is no coverage or no leaf.
    """
    h, w = leaf_mask.shape
    cell_h = max(h // grid_n, 1)
    cell_w = max(w // grid_n, 1)

    fractions = []
    for i in range(grid_n):
        for j in range(grid_n):
            y0 = i * cell_h
            y1 = y0 + cell_h if i < grid_n - 1 else h
            x0 = j * cell_w
            x1 = x0 + cell_w if j < grid_n - 1 else w

            leaf_cell = leaf_mask[y0:y1, x0:x1]
            contact_cell = contacted_mask[y0:y1, x0:x1]

            leaf_pix = int(np.count_nonzero(leaf_cell))
            if leaf_pix == 0:
                continue  # outside the leaf — skip this cell

            contact_pix = int(np.count_nonzero(contact_cell))
            fractions.append(contact_pix / leaf_pix)

    if not fractions or sum(fractions) == 0.0:
        return 0.0

    fracs = np.array(fractions, dtype=np.float64)
    total = fracs.sum()
    if total == 0.0:
        return 0.0

    # Normalise to a probability distribution and compute Shannon entropy
    probs = fracs / total
    # Avoid log(0)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))

    # Normalise by maximum possible entropy (uniform distribution)
    n_cells = len(fractions)
    max_entropy = np.log(n_cells) if n_cells > 1 else 1.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def effectiveness_score(coverage_pct: float, uniformity: float) -> float:
    """
    Overall effectiveness score on a 0–100 scale.

    Rewards both high coverage AND even distribution:
        score = coverage_pct × uniformity

    A leaf with 60 % coverage but very uneven distribution scores lower
    than one with 50 % coverage that is uniformly spread.
    """
    return float(np.clip(coverage_pct * uniformity, 0.0, 100.0))
