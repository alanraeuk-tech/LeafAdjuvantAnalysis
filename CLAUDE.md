# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **OS**: Windows 11, Claude Code via Git Bash (use Unix paths/syntax)
- **Python**: 3.14 ARM64 at `C:/Users/alanr/AppData/Local/Programs/Python/Python314-arm64/python.exe`
- **opencv-python is not available** — no ARM64 wheel exists for Python 3.14. Use `scikit-image` for all image processing instead.

## Running the Application

```bash
# Desktop GUI (tkinter)
python main.py

# Web interface (Streamlit)
streamlit run app.py
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Smoke Test

The `.claude/settings.local.json` defines an inline integration test that creates a synthetic leaf image and runs `analyse_image()`. It validates coverage, uniformity, and effectiveness score output. No formal test framework exists.

## Architecture

The application is split into four layers with clean separation:

- **`analyser.py`** — Pure image analysis pipeline. Takes numpy `uint8` RGB arrays, returns an `AnalysisResult` dataclass. No I/O or GUI.
- **`metrics.py`** — Pure metric calculations (coverage, uniformity via Shannon entropy, effectiveness score). Called by both UIs.
- **`main.py`** — tkinter desktop GUI. Single `LeafAnalyserApp` class. 3-column layout (image 1 | image 2 | results). Debounces parameter changes by 600 ms before re-running analysis.
- **`app.py`** — Streamlit web interface. Mirrors the GUI's functionality for browser/share.streamlit.io deployment.

### Analysis Pipeline (analyser.py)

`analyse_image()` is the top-level entry point. Internally:
1. `segment_leaf()` — HSV colour threshold (hue 0.18–0.55, sat/val ≥ 0.10) + morphological close/open; keeps largest connected component.
2. `detect_wet_areas()` — LAB colour space; uses 75th-percentile lightness of the leaf as "dry" reference; marks pixels deviating > threshold.
3. `analyse_blobs()` — Connected components, minimum area 10 px; extracts count, area, diameter.
4. `size_histogram()` — Bins blobs: small < 20 px, medium 20–49 px, large ≥ 50 px.
5. `create_overlay()` — PIL rendering: background dimmed to 40%, wet areas blended with semi-transparent green, red contours around blobs.

### User-Adjustable Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Threshold | 5–100 | 30 | LAB lightness deviation for spray detection |
| Grid Size | 4–16 (even) | 8 | Cell resolution for uniformity (Shannon entropy) |
