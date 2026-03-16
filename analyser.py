"""
analyser.py — Image-analysis pipeline for the Leaf Adjuvant Analyser.

Uses scikit-image (skimage) + numpy + Pillow instead of OpenCV, because
opencv-python has no pre-built wheel for Python 3.14 ARM64 as of mid-2025.

All functions are pure (no GUI, no I/O).  The top-level entry point is
analyse_image(), which accepts an RGB uint8 numpy array and returns an
AnalysisResult.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, morphology, measure


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    leaf_mask:      np.ndarray           # bool — leaf region
    contacted_mask: np.ndarray           # bool — spray-contacted region
    overlay_image:  np.ndarray           # RGB uint8 — visualisation
    blob_stats:     List[Tuple[int, float]] = field(default_factory=list)
    # each entry: (area_px, equivalent_diameter_px)
    droplet_count:  int   = 0
    mean_diameter:  float = 0.0
    size_bins:      Tuple[int, int, int] = (0, 0, 0)  # small / medium / large


# ---------------------------------------------------------------------------
# Step 1 — Leaf segmentation
# ---------------------------------------------------------------------------

def segment_leaf(image_rgb: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask that isolates the leaf from the background.

    Strategy:
      1. Convert to HSV (skimage: all channels in [0, 1]).
      2. Threshold on green hue, non-trivial saturation and value.
      3. Morphological close then open to fill gaps and remove speckle.
      4. Keep only the largest connected component (the main leaf).
    """
    img_float = image_rgb.astype(np.float64) / 255.0
    hsv = color.rgb2hsv(img_float)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # skimage HSV: H in [0, 1], green ≈ 0.33
    # Range 0.18–0.55 covers yellow-green through teal-green
    mask = (
        (h >= 0.18) & (h <= 0.55) &
        (s >= 0.10) &
        (v >= 0.10)
    )

    # Morphological cleanup — fill holes, remove isolated specks
    disk_close = morphology.disk(15)
    disk_open  = morphology.disk(10)
    mask = morphology.closing(mask, disk_close)
    mask = morphology.opening(mask, disk_open)

    # Keep only the largest connected component
    labeled = measure.label(mask, connectivity=2)
    if labeled.max() == 0:
        # Nothing detected — use the full image so the pipeline doesn't crash
        return np.ones(mask.shape, dtype=bool)

    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return labeled == largest.label


# ---------------------------------------------------------------------------
# Step 2 — Wet-area / spray-contact detection
# ---------------------------------------------------------------------------

def detect_wet_areas(image_rgb: np.ndarray, leaf_mask: np.ndarray,
                     threshold: int = 30) -> np.ndarray:
    """
    Return a boolean mask of spray-contacted leaf pixels.

    Strategy:
      - Convert to LAB colour space (perceptually uniform).
        skimage LAB: L in [0, 100], A/B in [-128, 127].
      - Compute the 75th-percentile L value of leaf pixels as a proxy for
        "bare leaf" brightness.
      - Pixels whose L value deviates from that reference by more than the
        scaled threshold are classified as "contacted".
      - Light morphological open/close to clean up noise.

    threshold (int, 5–100): GUI slider value.  Mapped to LAB L units via
        lab_threshold = threshold × 100 / 255
    so that the default of 30 corresponds to ~11.8 LAB L units — equivalent
    to the cv2 LAB scale the design was originally specified around.
    """
    img_float = image_rgb.astype(np.float64) / 255.0
    lab = color.rgb2lab(img_float)
    l_channel = lab[:, :, 0]           # [0, 100]

    # Scale from slider range (0–100) to LAB L units
    lab_threshold = threshold * 100.0 / 255.0

    leaf_l = l_channel[leaf_mask]
    if leaf_l.size == 0:
        return np.zeros_like(leaf_mask, dtype=bool)

    bare_ref = float(np.percentile(leaf_l, 75))

    diff = np.abs(l_channel - bare_ref)
    contacted = (diff > lab_threshold) & leaf_mask

    # Morphological cleanup
    disk_small = morphology.disk(3)
    contacted = morphology.opening(contacted, disk_small)
    contacted = morphology.closing(contacted, disk_small)

    return contacted


# ---------------------------------------------------------------------------
# Step 3 — Blob (droplet) analysis
# ---------------------------------------------------------------------------

def analyse_blobs(contacted_mask: np.ndarray,
                  min_area: int = 10
                  ) -> Tuple[List[Tuple[int, float]], int, float]:
    """
    Find connected components (droplets) in the contacted mask.

    Returns:
        blob_stats     — list of (area_px, equiv_diameter_px)
        droplet_count  — number of blobs above min_area
        mean_diameter  — mean equivalent circular diameter
    """
    labeled = measure.label(contacted_mask, connectivity=2)
    regions = measure.regionprops(labeled)

    blob_stats: List[Tuple[int, float]] = []
    for region in regions:
        if region.area < min_area:
            continue
        diameter = float(region.equivalent_diameter_area)
        blob_stats.append((int(region.area), diameter))

    if not blob_stats:
        return blob_stats, 0, 0.0

    droplet_count = len(blob_stats)
    mean_diameter = float(np.mean([d for _, d in blob_stats]))
    return blob_stats, droplet_count, mean_diameter


def size_histogram(blob_stats: List[Tuple[int, float]]) -> Tuple[int, int, int]:
    """
    Bin blobs into small / medium / large by equivalent diameter.

        small  : diameter <  20 px
        medium : 20 ≤ diameter < 50 px
        large  : diameter ≥ 50 px
    """
    if not blob_stats:
        return (0, 0, 0)
    diameters = [d for _, d in blob_stats]
    small  = sum(1 for d in diameters if d < 20)
    medium = sum(1 for d in diameters if 20 <= d < 50)
    large  = sum(1 for d in diameters if d >= 50)
    return (small, medium, large)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_overlay(image_rgb: np.ndarray, contacted_mask: np.ndarray,
                   leaf_mask: np.ndarray) -> np.ndarray:
    """
    Render a visual overlay (RGB uint8):
      - Dimmed non-leaf background so the leaf stands out.
      - Semi-transparent green fill over contacted areas.
      - Red contour lines around individual droplet blobs.
    """
    base = image_rgb.astype(np.float64)

    # Dim the background
    bg = ~leaf_mask
    base[bg] *= 0.40

    # Semi-transparent green for contacted region
    green = np.zeros_like(base)
    green[contacted_mask, 1] = 200.0   # G channel in RGB
    blended = base * 0.70 + green * 0.30
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Red contours drawn with PIL
    pil = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil)

    contours = measure.find_contours(contacted_mask.astype(np.uint8), 0.5)
    for contour in contours:
        # contour coords are (row, col) → convert to PIL (x=col, y=row)
        pts = [(float(c[1]), float(c[0])) for c in contour]
        if len(pts) >= 2:
            # Close the contour
            draw.line(pts + [pts[0]], fill=(220, 0, 0), width=1)

    return np.array(pil)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def analyse_image(image_rgb: np.ndarray, threshold: int = 30) -> AnalysisResult:
    """
    Full analysis pipeline.

    image_rgb: RGB uint8 numpy array (H × W × 3).
    threshold: wet-area detection sensitivity (GUI slider, 5–100).
    """
    leaf_mask      = segment_leaf(image_rgb)
    contacted_mask = detect_wet_areas(image_rgb, leaf_mask, threshold)
    blob_stats, droplet_count, mean_diameter = analyse_blobs(contacted_mask)
    bins           = size_histogram(blob_stats)
    overlay        = create_overlay(image_rgb, contacted_mask, leaf_mask)

    return AnalysisResult(
        leaf_mask      = leaf_mask,
        contacted_mask = contacted_mask,
        overlay_image  = overlay,
        blob_stats     = blob_stats,
        droplet_count  = droplet_count,
        mean_diameter  = mean_diameter,
        size_bins      = bins,
    )
