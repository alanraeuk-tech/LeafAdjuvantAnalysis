# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Streamlit app that analyses photos/scans of water-sensitive paper (WSP) to compare spray deposits with and without an adjuvant. Despite the repo name, it analyses WSP cards, not leaves. User guide: `GUIDE.md`.

## Environment

- **OS**: Windows 11, Claude Code via Git Bash (use Unix paths/syntax)
- **Python**: 3.14 ARM64 at `C:/Users/alanr/AppData/Local/Programs/Python/Python314-arm64/python.exe`
- **opencv-python is not available** (no ARM64 wheel for Python 3.14). Use `scikit-image` / `scipy.ndimage` for image processing.

## Commands

```bash
pip install -r requirements.txt
streamlit run app.py          # the app
python -m pytest              # tests (synthetic cards with known answers)
```

## Architecture

- **`analyser.py`**: pure image pipeline. RGB uint8 array in, `AnalysisResult` out (stain mask, droplet labels, diameters in px, overlay). Also `calibrate_from_card()` which gives µm/px from the card size. No I/O, no UI.
- **`metrics.py`**: turns an `AnalysisResult` + `Calibration` into `DepositMetrics` (coverage, density/cm², Dv0.1/0.5/0.9, NMD, span, merged %, coverage CV, warnings). Pure.
- **`app.py`**: Streamlit UI. Caches analysis on (image bytes, detection params) so reporting-only changes are instant.
- **`tests/test_pipeline.py`**: renders synthetic cards with known circles and checks counts, sizes, splitting, edge handling, metric maths.

### Pipeline (`analyse_image`)

1. `trim_border`: drop a 3 px frame.
2. `detect_stains`: LAB colour distance (ΔE) from the median paper colour > threshold, plus small isolated specks found against a local background (grey closing). Holes ≤ 4 px filled.
3. `split_droplets`: remove objects under the noise floor, watershed on the smoothed distance transform with h-maxima seeds; `_merge_close_markers` joins seeds closer than 0.75 × local radius; `_absorb_small_fragments` folds slivers back in.
4. `create_overlay`: green tint on stains, red per-pixel boundaries (no polyline closing, so no stray edge lines).

### Conventions

- All sizes in the UI and metrics are µm. Pixels only inside `analyser.py`.
- Sizes are **stain** diameters unless the user sets a spread factor.
- Don't reintroduce a single "effectiveness score": it was coverage × an entropy measure that barely varied, and misled more than it helped.
- Warnings (resolution, crop, high coverage) live in `metrics.summarise` so the app and report share them.
