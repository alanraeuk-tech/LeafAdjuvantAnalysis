"""
analyser.py — Image-analysis pipeline for the WSP Adjuvant Analyser.

Uses scikit-image (skimage) + numpy + Pillow instead of OpenCV, because
opencv-python has no pre-built wheel for Python 3.14 ARM64 as of mid-2025.

All functions are pure (no GUI, no I/O).  The top-level entry point is
analyse_image(), which accepts an RGB uint8 numpy array and returns an
AnalysisResult.

Input type: water-sensitive paper (WSP) photographed in a controlled dark box.
WSP turns blue where spray droplets land; uncontacted areas remain bright yellow.
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
# Step 1 — Paper segmentation
# ---------------------------------------------------------------------------

def segment_paper(image_rgb: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask covering the entire WSP image frame.

    WSP images are taken in a controlled dark box; the whole frame is paper,
    so no background segmentation is needed.
    """
    h, w = image_rgb.shape[:2]
    return np.ones((h, w), dtype=bool)


# ---------------------------------------------------------------------------
# Step 2 — Blue-droplet detection
# ---------------------------------------------------------------------------

def detect_droplets(image_rgb: np.ndarray, paper_mask: np.ndarray,
                    threshold: int = 30) -> np.ndarray:
    """
    Return a boolean mask of blue spray-droplet pixels on WSP.

    Strategy:
      - Convert to LAB colour space.
      - WSP paper is yellow (high b*); spray droplets shift toward blue/dark
        (lower b*), whether vivid blue or near-black concentrated spots.
      - Use the 90th-percentile b* of paper pixels as the "dry yellow" reference.
      - Mark any pixel whose b* falls more than `threshold` units below that
        reference as contacted.  Lower threshold = more sensitive.
      - Small morphological closing (disk 1) to fill single-pixel gaps inside
        droplets without destroying small spots.
    """
    img_float = image_rgb.astype(np.float64) / 255.0
    lab = color.rgb2lab(img_float)
    b_star = lab[..., 2]   # positive = yellow, negative/low = blue

    # Reference: the bright yellow paper colour (top 90th percentile of b*)
    paper_b_ref = np.percentile(b_star[paper_mask], 90)

    contacted = (b_star < paper_b_ref - threshold) & paper_mask

    # Closing only: fill small gaps without eroding small spots
    contacted = morphology.closing(contacted, morphology.disk(1))

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
                   paper_mask: np.ndarray) -> np.ndarray:
    """
    Render a visual overlay (RGB uint8):
      - Original yellow paper background (no dimming — entire frame is paper).
      - Semi-transparent green fill over detected blue droplet areas.
      - Red contour lines around individual droplet blobs.
    """
    base = image_rgb.astype(np.float64)

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
    Full WSP analysis pipeline.

    image_rgb: RGB uint8 numpy array (H × W × 3).
    threshold: blue-droplet detection sensitivity — controls minimum HSV
               saturation (GUI slider, 5–100).
    """
    paper_mask     = segment_paper(image_rgb)
    contacted_mask = detect_droplets(image_rgb, paper_mask, threshold)
    blob_stats, droplet_count, mean_diameter = analyse_blobs(contacted_mask)
    bins           = size_histogram(blob_stats)
    overlay        = create_overlay(image_rgb, contacted_mask, paper_mask)

    return AnalysisResult(
        leaf_mask      = paper_mask,
        contacted_mask = contacted_mask,
        overlay_image  = overlay,
        blob_stats     = blob_stats,
        droplet_count  = droplet_count,
        mean_diameter  = mean_diameter,
        size_bins      = bins,
    )
