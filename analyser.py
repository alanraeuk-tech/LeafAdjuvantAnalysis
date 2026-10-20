"""
analyser.py — Image-analysis pipeline for water-sensitive paper (WSP).

WSP is yellow; it turns blue/dark wherever a spray droplet lands.  This module
finds those stains, separates stains that have merged, and returns per-droplet
sizes in pixels.  Converting pixels to microns, and turning sizes into
agronomic metrics, happens in metrics.py.

Uses scikit-image rather than OpenCV (no opencv wheel for Python 3.14 ARM64).
All functions are pure: numpy in, numpy/dataclasses out.  No GUI, no I/O.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation


# Standard WSP card (e.g. Syngenta / TeeJet 20301-2N) is 26 x 76 mm.
DEFAULT_CARD_WIDTH_MM = 26.0
DEFAULT_CARD_HEIGHT_MM = 76.0

# Colour distance (CIE76 delta E) from the paper colour above which a pixel
# counts as stained.  Roughly the halfway point between yellow paper and a
# small dark stain, so stain edges are measured at about half intensity.
DEFAULT_THRESHOLD = 35

# Stains smaller than this (equivalent diameter, pixels) are treated as noise.
# A 2 px stain can be counted but not sized well; see metrics.reliable_min_um.
DEFAULT_MIN_DIAMETER_PX = 2.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    um_per_px: float          # microns per pixel
    aspect_mismatch: float    # 0 = image aspect matches the card exactly

    @property
    def px_per_mm(self) -> float:
        return 1000.0 / self.um_per_px


def calibrate_from_card(width_px: int, height_px: int,
                        card_width_mm: float = DEFAULT_CARD_WIDTH_MM,
                        card_height_mm: float = DEFAULT_CARD_HEIGHT_MM
                        ) -> Calibration:
    """
    Work out the pixel size, assuming the image is cropped to the card edges.

    Uses the mean of the horizontal and vertical scales.  aspect_mismatch is
    the relative difference between them: anything over ~0.05 means the photo
    is not cropped to the card, is skewed, or the card size is wrong, and the
    micron figures should not be trusted.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("image dimensions must be positive")
    sx = card_width_mm * 1000.0 / width_px
    sy = card_height_mm * 1000.0 / height_px
    mismatch = abs(sx - sy) / ((sx + sy) / 2.0)
    return Calibration(um_per_px=(sx + sy) / 2.0, aspect_mismatch=mismatch)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    image_rgb:       np.ndarray   # analysed image (after border trim)
    stain_mask:      np.ndarray   # bool, stained pixels
    droplet_labels:  np.ndarray   # int, one label per separated droplet
    diameters_px:    np.ndarray   # equivalent diameter of each droplet
    stain_count:     int          # connected stains before splitting
    merged_fraction: float        # share of stained area in merged stains
    paper_lab:       np.ndarray   # LAB colour used as the dry-paper reference
    overlay_image:   np.ndarray   # RGB uint8 visualisation

    @property
    def droplet_count(self) -> int:
        return int(len(self.diameters_px))

    @property
    def paper_area_px(self) -> int:
        return int(self.stain_mask.size)


# ---------------------------------------------------------------------------
# Step 1 — find stained pixels
# ---------------------------------------------------------------------------

def trim_border(image_rgb: np.ndarray, border_px: int) -> np.ndarray:
    """Drop a thin frame: scans and crops often leave a white or dark edge."""
    if border_px <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if h <= 2 * border_px or w <= 2 * border_px:
        return image_rgb
    return image_rgb[border_px:h - border_px, border_px:w - border_px]


def estimate_paper_colour(lab: np.ndarray) -> np.ndarray:
    """
    Median LAB colour of the clearly-yellow pixels.

    Otsu on b* (the yellow/blue axis) splits yellow paper from blue stain
    well enough to pick reference pixels, even at high coverage.
    """
    b_star = lab[..., 2]
    if np.ptp(b_star) < 1e-6:
        return np.median(lab.reshape(-1, 3), axis=0)
    paper = b_star > filters.threshold_otsu(b_star)
    return np.median(lab[paper], axis=0)


def detect_stains(image_rgb: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
                  local_window_px: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (stain_mask, paper_lab).

    Two passes:

    1. Global: a pixel is stained if its colour is more than `threshold`
       delta-E units from the card's paper colour.  Full colour distance
       (not just yellow/blue) catches dark stains as well as blue ones.
       This sets the size of the larger stains.

    2. Local: small, faint stains (fine droplets, blurred by low-resolution
       photos) can sit under the global threshold.  Each pixel is compared
       with the paper colour in its own neighbourhood instead.  Only small,
       isolated specks are taken from this pass: near a large stain the
       local comparison picks up its blurred halo and would inflate it.
    """
    lab = color.rgb2lab(image_rgb.astype(np.float64) / 255.0)
    paper_lab = estimate_paper_colour(lab)

    global_mask = np.linalg.norm(lab - paper_lab, axis=-1) > threshold
    global_mask = _fill_small_holes(global_mask, max_hole_px=4)

    if local_window_px >= 3:
        # Paper is bright and yellow: a grey closing removes dark specks
        # smaller than the window and leaves the local paper colour.
        bg = np.stack([
            ndi.grey_closing(lab[..., 0], size=local_window_px),
            ndi.uniform_filter(lab[..., 1], size=local_window_px),
            ndi.grey_closing(lab[..., 2], size=local_window_px),
        ], axis=-1)
        local_mask = np.linalg.norm(lab - bg, axis=-1) > threshold
        specks = measure.label(local_mask, connectivity=2)
        sizes = np.bincount(specks.ravel())
        max_speck = int(local_window_px ** 2 / 8)
        near_global = ndi.binary_dilation(global_mask, iterations=2)
        touching = np.unique(specks[near_global])
        keep = sizes <= max_speck
        keep[0] = False
        keep[touching] = False
        global_mask |= keep[specks]

    return global_mask, paper_lab


def _fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    holes = ~mask
    lab = measure.label(holes, connectivity=1)
    sizes = np.bincount(lab.ravel())
    small = sizes <= max_hole_px
    small[0] = False
    return mask | small[lab]


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    lab = measure.label(mask, connectivity=2)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------------------
# Step 2 — separate merged stains into droplets
# ---------------------------------------------------------------------------

def split_droplets(mask: np.ndarray, min_diameter_px: float,
                   split: bool = True) -> tuple[np.ndarray, int, float]:
    """
    Label individual droplets.

    Overlapping round stains form a single blob with a "waist".  A watershed
    on the distance transform cuts at the waists.  The h-maxima marker step
    stops irregular single stains being shredded: a bulge has to stand at
    least ~1.5 px proud of its neighbour to count as a separate droplet.

    Returns (labels, raw_stain_count, merged_area_fraction).
    """
    min_area = max(int(round(np.pi * (min_diameter_px / 2.0) ** 2)), 1)
    mask = _remove_small_objects(mask, min_area)

    raw = measure.label(mask, connectivity=2)
    raw_count = int(raw.max())
    if raw_count == 0 or not split:
        return raw, raw_count, 0.0

    dist = ndi.distance_transform_edt(mask)
    dist = filters.gaussian(dist, sigma=0.7, preserve_range=True)
    peaks = morphology.h_maxima(dist, 1.5)
    markers = _merge_close_markers(measure.label(peaks), dist)
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # Blobs that held no marker (very thin shapes) keep their raw label.
    missing = mask & (labels == 0)
    if missing.any():
        extra = measure.label(missing, connectivity=2)
        labels[missing] = extra[missing] + labels.max()

    # Drop fragments the split produced that fall under the noise floor,
    # by merging them back into an adjacent droplet.
    labels = _absorb_small_fragments(labels, min_area)

    # Share of stained area sitting in raw blobs that split into 2+ droplets.
    merged_px = 0
    for region in measure.regionprops(raw):
        sl = region.slice
        inside = labels[sl][region.image]
        if len(np.unique(inside[inside > 0])) > 1:
            merged_px += region.area
    merged_fraction = merged_px / max(int(mask.sum()), 1)

    return labels, raw_count, float(merged_fraction)


def _merge_close_markers(markers: np.ndarray, dist: np.ndarray,
                         ratio: float = 0.75) -> np.ndarray:
    """
    Join watershed seeds that are too close to be separate droplets.

    Two overlapping round stains have centres at least most of a radius
    apart.  Seeds closer than `ratio` x the smaller local radius are pixel
    noise on one stain's distance ridge, so they get the same label.
    """
    n = int(markers.max())
    if n < 2:
        return markers
    idx = np.arange(1, n + 1)
    centres = np.array(ndi.center_of_mass(np.ones_like(dist), markers, idx))
    radii = np.array(ndi.maximum(dist, markers, idx))

    parent = list(range(n + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Neighbouring seeds only: compare each seed with those within the
    # largest possible merge distance (cheap even with thousands of seeds).
    order = np.argsort(centres[:, 0])
    reach = ratio * radii.max()
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1:]:
            if centres[b, 0] - centres[a, 0] > reach:
                break
            gap = np.hypot(*(centres[a] - centres[b]))
            if gap < ratio * min(radii[a], radii[b]):
                parent[find(a + 1)] = find(b + 1)

    lut = np.array([find(i) for i in range(n + 1)])
    return lut[markers]


def _absorb_small_fragments(labels: np.ndarray, min_area: int) -> np.ndarray:
    labels = labels.copy()
    areas = np.bincount(labels.ravel())
    slices = ndi.find_objects(labels)
    h, w = labels.shape
    for lab in np.where((areas > 0) & (areas < min_area))[0]:
        if lab == 0 or slices[lab - 1] is None:
            continue
        sy, sx = slices[lab - 1]
        y0, y1 = max(sy.start - 1, 0), min(sy.stop + 1, h)
        x0, x1 = max(sx.start - 1, 0), min(sx.stop + 1, w)
        win = labels[y0:y1, x0:x1]
        frag = win == lab
        ring = ndi.binary_dilation(frag) & ~frag
        neighbours = win[ring]
        neighbours = neighbours[(neighbours > 0) & (neighbours != lab)]
        win[frag] = np.bincount(neighbours).argmax() if neighbours.size else 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_overlay(image_rgb: np.ndarray, stain_mask: np.ndarray,
                   labels: np.ndarray) -> np.ndarray:
    """
    Green tint over stained area, red outline round each droplet (including
    the split lines between merged droplets).  Drawn per-pixel, so stains
    touching the image edge can't produce stray lines.
    """
    out = image_rgb.astype(np.float64)
    green = np.array([0.0, 200.0, 0.0])
    out[stain_mask] = out[stain_mask] * 0.6 + green * 0.4
    edges = segmentation.find_boundaries(labels, mode="inner")
    out[edges] = [220.0, 0.0, 0.0]
    return np.clip(out, 0, 255).astype(np.uint8)


def upscale_for_display(image_rgb: np.ndarray, min_width: int = 500
                        ) -> np.ndarray:
    """Nearest-neighbour upscale so thin outlines stay visible on screen."""
    h, w = image_rgb.shape[:2]
    if w >= min_width:
        return image_rgb
    f = int(np.ceil(min_width / w))
    return np.array(Image.fromarray(image_rgb).resize((w * f, h * f),
                                                      Image.NEAREST))


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def analyse_image(image_rgb: np.ndarray,
                  threshold: float = DEFAULT_THRESHOLD,
                  min_diameter_px: float = DEFAULT_MIN_DIAMETER_PX,
                  split_merged: bool = True,
                  border_px: int = 3) -> AnalysisResult:
    """
    Full WSP pipeline.

    image_rgb:       RGB uint8 array, cropped to the card.
    threshold:       colour distance (delta E) from paper to count as stain.
    min_diameter_px: stains smaller than this are ignored as noise.
    split_merged:    separate overlapping stains with a watershed.
    border_px:       frame to trim before analysis.
    """
    img = trim_border(image_rgb, border_px)
    mask, paper_lab = detect_stains(img, threshold)
    labels, raw_count, merged = split_droplets(mask, min_diameter_px,
                                               split_merged)
    mask = labels > 0

    if labels.max() > 0:
        areas = np.bincount(labels.ravel())[1:]
        areas = areas[areas > 0]
        diameters = np.sqrt(4.0 * areas / np.pi)
    else:
        diameters = np.zeros(0)

    return AnalysisResult(
        image_rgb=img,
        stain_mask=mask,
        droplet_labels=labels,
        diameters_px=diameters.astype(np.float64),
        stain_count=raw_count,
        merged_fraction=merged,
        paper_lab=paper_lab,
        overlay_image=create_overlay(img, mask, labels),
    )
