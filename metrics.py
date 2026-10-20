"""
metrics.py — Spray-deposit metrics from an AnalysisResult.

All sizes are reported in microns using a Calibration.  The metrics follow
the usual conventions for water-sensitive paper analysis:

  coverage %          stained area / card area
  density             droplets per cm^2 of card
  Dv0.1 / Dv0.5 / Dv0.9
                      diameters below which 10 / 50 / 90 % of the spray
                      *volume* sits.  Dv0.5 is the volume median diameter
                      (VMD), the headline droplet-size figure.
  NMD                 number median diameter (half the droplets are smaller)
  relative span       (Dv0.9 - Dv0.1) / Dv0.5; lower = more uniform sizes
  coverage CV %       how unevenly coverage is spread across the card;
                      lower = more even

Sizes are *stain* diameters.  A droplet spreads as it lands, so the stain is
larger than the droplet that made it.  Pass a spread_factor > 1 to convert
(droplet = stain / spread_factor); the right factor depends on the paper and
the liquid, and an adjuvant changes it, so by default stains are reported
as-is.
"""

from dataclasses import dataclass

import numpy as np

from analyser import AnalysisResult, Calibration

# Below this the image cannot resolve fine droplets.  ~600 dpi scan.
RECOMMENDED_MAX_UM_PER_PX = 45.0
# Above this, overlapping stains make droplet sizing unreliable.
HIGH_COVERAGE_PCT = 30.0
# Stains need roughly this many pixels across to be sized, not just counted.
MIN_SIZEABLE_PX = 3.0


@dataclass
class DepositMetrics:
    coverage_pct:      float
    droplet_count:     int
    stain_count:       int
    density_per_cm2:   float
    dv10_um:           float
    dv50_um:           float
    dv90_um:           float
    nmd_um:            float
    relative_span:     float
    merged_pct:        float
    coverage_cv_pct:   float
    um_per_px:         float
    smallest_sized_um: float
    warnings:          list


def coverage_percent(stain_mask: np.ndarray) -> float:
    if stain_mask.size == 0:
        return 0.0
    return float(np.count_nonzero(stain_mask) / stain_mask.size * 100.0)


def density_per_cm2(count: int, area_px: int, um_per_px: float) -> float:
    area_cm2 = area_px * (um_per_px / 10_000.0) ** 2
    return float(count / area_cm2) if area_cm2 > 0 else 0.0


def volume_diameters(diameters: np.ndarray,
                     fractions=(0.1, 0.5, 0.9)) -> tuple:
    """
    Volume-weighted percentile diameters (Dv0.1, Dv0.5, Dv0.9 by default).

    Each droplet's volume is proportional to d^3.  Sort by size, build the
    cumulative volume curve, and read off where it crosses each fraction.
    """
    d = np.sort(np.asarray(diameters, dtype=np.float64))
    if d.size == 0:
        return tuple(0.0 for _ in fractions)
    if d.size == 1:
        return tuple(float(d[0]) for _ in fractions)
    vol = d ** 3
    # Mid-point cumulative fraction for each droplet, so a single dominant
    # droplet doesn't pin every percentile to the largest size.
    cum = (np.cumsum(vol) - vol / 2.0) / vol.sum()
    return tuple(float(np.interp(f, cum, d)) for f in fractions)


def cumulative_volume_curve(diameters: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """(sorted diameters, cumulative volume %) for plotting."""
    d = np.sort(np.asarray(diameters, dtype=np.float64))
    if d.size == 0:
        return d, d
    vol = d ** 3
    return d, np.cumsum(vol) / vol.sum() * 100.0


def coverage_cv(stain_mask: np.ndarray, cell_px: int) -> float:
    """
    Coefficient of variation (%) of coverage across square cells.

    Partial cells at the right/bottom edge are dropped so every cell has the
    same area.  Returns 0 when there's no coverage or fewer than two cells.
    """
    cell_px = max(int(cell_px), 1)
    h, w = stain_mask.shape
    ny, nx = h // cell_px, w // cell_px
    if ny * nx < 2:
        return 0.0
    m = stain_mask[:ny * cell_px, :nx * cell_px].astype(np.float64)
    cells = m.reshape(ny, cell_px, nx, cell_px).mean(axis=(1, 3))
    mean = cells.mean()
    if mean == 0:
        return 0.0
    return float(cells.std() / mean * 100.0)


def summarise(result: AnalysisResult, calibration: Calibration,
              cell_mm: float = 5.0, spread_factor: float = 1.0
              ) -> DepositMetrics:
    um = calibration.um_per_px
    sf = max(float(spread_factor), 1e-6)
    d_um = result.diameters_px * um / sf

    dv10, dv50, dv90 = volume_diameters(d_um)
    nmd = float(np.median(d_um)) if d_um.size else 0.0
    span = (dv90 - dv10) / dv50 if dv50 > 0 else 0.0
    cov = coverage_percent(result.stain_mask)
    cell_px = int(round(cell_mm * 1000.0 / um))

    warnings = []
    if calibration.aspect_mismatch > 0.08:
        warnings.append(
            f"Image shape doesn't match the card size "
            f"({calibration.aspect_mismatch:.0%} off). Crop the photo to "
            f"the card edges or check the card dimensions, otherwise the "
            f"micron figures are off.")
    if um > RECOMMENDED_MAX_UM_PER_PX:
        warnings.append(
            f"Low resolution: {um:.0f} µm per pixel. Stains under about "
            f"{MIN_SIZEABLE_PX * um / sf:.0f} µm can't be sized and the "
            f"finest ones are missed entirely, so Dv0.1 and droplet counts "
            f"are understated. Scan the card at 600 dpi or better.")
    if cov > HIGH_COVERAGE_PCT:
        warnings.append(
            f"Coverage is {cov:.0f}%. With this much overlap, droplet "
            f"counts and sizes are estimates; coverage itself is still "
            f"reliable.")

    return DepositMetrics(
        coverage_pct=cov,
        droplet_count=result.droplet_count,
        stain_count=result.stain_count,
        density_per_cm2=density_per_cm2(result.droplet_count,
                                        result.paper_area_px, um),
        dv10_um=dv10, dv50_um=dv50, dv90_um=dv90,
        nmd_um=nmd,
        relative_span=float(span),
        merged_pct=result.merged_fraction * 100.0,
        coverage_cv_pct=coverage_cv(result.stain_mask, cell_px),
        um_per_px=um,
        smallest_sized_um=MIN_SIZEABLE_PX * um / sf,
        warnings=warnings,
    )
