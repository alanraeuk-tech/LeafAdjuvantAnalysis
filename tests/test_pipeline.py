"""
Tests against synthetic water-sensitive paper cards with known droplets.

Run with:  python -m pytest
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from analyser import analyse_image, calibrate_from_card
from metrics import (coverage_cv, density_per_cm2, summarise,
                     volume_diameters)

PAPER = np.array([245, 230, 40], dtype=np.float64)    # WSP yellow
STAIN = np.array([30, 40, 160], dtype=np.float64)     # spray blue


def make_card(circles, shape=(760, 260), blur=0.8, stain=STAIN, strength=1.0):
    """Render circles [(row, col, radius_px), ...] onto a yellow card."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ink = np.zeros(shape)
    for r0, c0, rad in circles:
        ink[(yy - r0) ** 2 + (xx - c0) ** 2 <= rad ** 2] = strength
    if blur:
        ink = ndi.gaussian_filter(ink, blur)
    img = PAPER * (1 - ink[..., None]) + stain * ink[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# 26 x 76 mm card at 100 um/px
CAL = calibrate_from_card(260, 760)


def test_calibration_standard_card():
    assert CAL.um_per_px == pytest.approx(100.0, rel=0.01)
    assert CAL.aspect_mismatch < 0.01


def test_calibration_flags_uncropped_photo():
    cal = calibrate_from_card(400, 760)   # far too wide for a 26 x 76 card
    assert cal.aspect_mismatch > 0.3


def test_blank_card():
    res = analyse_image(make_card([]))
    m = summarise(res, CAL)
    assert res.droplet_count == 0
    assert m.coverage_pct == pytest.approx(0.0, abs=0.01)
    assert m.dv50_um == 0.0


def test_isolated_droplets_counted_and_sized():
    rng = np.random.default_rng(1)
    circles = []
    for row in range(40, 720, 60):
        for col in range(40, 230, 60):
            circles.append((row + rng.integers(-5, 5),
                            col + rng.integers(-5, 5), 8))
    res = analyse_image(make_card(circles))
    assert res.droplet_count == len(circles)
    # 8 px radius = 16 px = 1600 um diameter
    d_um = res.diameters_px * CAL.um_per_px
    assert np.median(d_um) == pytest.approx(1600, rel=0.1)

    true_cov = len(circles) * np.pi * 8 ** 2 / (754 * 254) * 100
    assert summarise(res, CAL).coverage_pct == pytest.approx(true_cov,
                                                             rel=0.12)


def test_overlapping_pair_is_split():
    card = make_card([(380, 120, 12), (380, 140, 12)])
    res = analyse_image(card)
    assert res.stain_count == 1
    assert res.droplet_count == 2
    assert res.merged_fraction == pytest.approx(1.0)


def test_split_can_be_turned_off():
    card = make_card([(380, 120, 12), (380, 140, 12)])
    res = analyse_image(card, split_merged=False)
    assert res.droplet_count == 1
    assert res.merged_fraction == 0.0


def test_irregular_single_stain_not_shredded():
    # A slightly elongated stain should stay one droplet.
    card = make_card([(380, 130, 12), (380, 136, 12)])
    res = analyse_image(card)
    assert res.droplet_count == 1


def test_faint_small_speck_detected():
    # Fine droplet blurred to a pale smudge, next to a normal one.
    card = make_card([(200, 130, 10)])
    speck = make_card([(500, 130, 1.6)], strength=0.9)
    card = np.minimum(card, speck)
    res = analyse_image(card)
    assert res.droplet_count == 2


def test_edge_droplet_draws_no_stray_lines():
    card = make_card([(0, 130, 15), (760, 50, 15)])
    res = analyse_image(card)
    red = np.all(res.overlay_image == [220, 0, 0], axis=-1)
    assert red.any()
    near_stain = ndi.binary_dilation(res.stain_mask, iterations=2)
    assert not (red & ~near_stain).any()


def test_dark_stain_detected_not_just_blue():
    card = make_card([(380, 130, 10)], stain=np.array([40, 45, 30.0]))
    assert analyse_image(card).droplet_count == 1


def test_volume_diameters_equal_sizes():
    dv10, dv50, dv90 = volume_diameters(np.full(50, 300.0))
    assert dv10 == dv50 == dv90 == pytest.approx(300.0)


def test_volume_median_weighted_by_volume():
    # 99 droplets of 100 um plus one of 1000 um: the big one holds ~91 % of
    # the volume, so the VMD sits far above the number median.
    d = np.array([100.0] * 99 + [1000.0])
    _, dv50, _ = volume_diameters(d)
    assert dv50 > 500
    assert np.median(d) == 100


def test_density_per_cm2():
    # 1 cm^2 = 100 x 100 px at 100 um/px
    assert density_per_cm2(25, 100 * 100, 100.0) == pytest.approx(25.0)


def test_coverage_cv_detects_clustering():
    even = np.zeros((100, 100), bool)
    even[::10, ::10] = True
    clustered = np.zeros((100, 100), bool)
    clustered[:10, :10] = True
    assert coverage_cv(even, 20) == pytest.approx(0.0, abs=1e-9)
    assert coverage_cv(clustered, 20) > 100


def test_low_resolution_warning():
    res = analyse_image(make_card([(380, 130, 10)]))
    assert any("Low resolution" in w for w in summarise(res, CAL).warnings)
    fine = calibrate_from_card(2600, 7600)
    m = summarise(res, fine)
    assert not any("Low resolution" in w for w in m.warnings)


def test_spread_factor_scales_sizes():
    res = analyse_image(make_card([(380, 130, 10)]))
    a = summarise(res, CAL, spread_factor=1.0)
    b = summarise(res, CAL, spread_factor=2.0)
    assert b.dv50_um == pytest.approx(a.dv50_um / 2)
