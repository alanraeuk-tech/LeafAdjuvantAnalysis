# Leaf Adjuvant Analyser — User Guide

## What It Does

The Leaf Adjuvant Analyser is a desktop application for comparing how effectively an adjuvant (a spray additive) improves pesticide or liquid coverage on a leaf. You load two photos of a leaf — one sprayed **without** adjuvant and one sprayed **with** adjuvant — and the tool measures and visualises the spray contact on each.

---

## Running the App

```bash
cd LeafAdjuvantAnalysis
python main.py
```

---

## Window Layout

```
┌─────────────────────┬─────────────────────┬──────────────────┐
│  Image 1            │  Image 2            │  Results         │
│  (No Adjuvant)      │  (With Adjuvant)    │                  │
│                     │                     │  Metrics table   │
│  [Original photo]   │  [Original photo]   │                  │
│                     │                     │  Droplet size    │
│  [Analysis overlay] │  [Analysis overlay] │  bar chart       │
├─────────────────────┴─────────────────────┴──────────────────┤
│  Threshold: ──●──────  Grid size: [8]  [Analyse] [Save Report]│
└──────────────────────────────────────────────────────────────┘
```

The window has three columns and a bottom bar.

---

## Controls

### Load Image buttons

Each image column has a **Load Image 1** / **Load Image 2** button at the top.

- Opens a file browser — supports JPG, PNG, BMP, TIFF.
- The photo appears in the upper half of its column.
- The lower half shows "Not analysed yet" until you run analysis.
- You can reload a different image at any time.

---

### Threshold slider (bottom bar)

- **Range:** 5 – 100  (default: 30)
- Controls the **sensitivity** of spray-contact detection.
  - **Low value (e.g. 5–15):** Very sensitive — small brightness differences from the bare leaf are treated as spray contact. May over-detect on uneven or textured leaves.
  - **High value (e.g. 60–100):** Only strong, obvious wet patches are detected. May under-detect light or fine misting.
- The numeric readout next to the slider updates as you drag.
- If both images are loaded, analysis re-runs automatically ~0.6 s after you stop moving the slider.

**How it works internally:** The image is converted to LAB colour space. The 75th-percentile lightness of the leaf pixels is used as a "dry leaf" reference. Pixels whose lightness deviates from that reference by more than the threshold are marked as contacted.

---

### Grid size spinbox (bottom bar)

- **Range:** 4 – 16, step 2  (default: 8)
- Controls how finely the leaf is divided for the **Uniformity** calculation.
- The leaf is split into a *grid_n × grid_n* grid of cells. Uniformity measures how evenly spray contact is spread across those cells.
- **Smaller grid (e.g. 4):** Coarser spatial resolution — less sensitive to local clustering.
- **Larger grid (e.g. 16):** Finer resolution — better at detecting uneven patches but may be noisy on small leaves or low coverage.
- Changing the grid size re-triggers analysis automatically (same 0.6 s debounce as the threshold).

---

### Analyse button

Runs the full analysis pipeline on both loaded images using the current threshold and grid-size settings. Results update in the right-hand panel and the overlay images are drawn.

You can click this manually at any time. It is also triggered automatically when you adjust the threshold or grid size (with a short debounce delay).

---

### Save Report button

Exports a summary report as a **PNG** or **PDF** file. The report contains:

- Original and overlay images for both slots (side by side)
- A formatted metrics comparison table
- The droplet size distribution bar chart

You must run analysis before saving. A file-save dialog lets you choose the destination and format.

---

## Analysis Overlay

After analysis, the lower canvas in each image column shows the overlay:

| Visual element | Meaning |
|---|---|
| Dimmed area | Background (outside the leaf) |
| Normal leaf colour | Leaf area with no detected spray contact |
| Green tint | Spray-contacted region |
| Red outlines | Borders of individual spray droplet blobs |

---

## Metrics Table (Results panel)

| Metric | Description |
|---|---|
| **Coverage %** | Percentage of the leaf area showing spray contact |
| **Droplets** | Number of individual spray-contact blobs detected (min 10 px area each) |
| **Mean diam (px)** | Average equivalent circular diameter of the blobs, in pixels |
| **Uniformity** | 0–1 score: how evenly spray is spread across the leaf (1 = perfectly even) |
| **Eff. Score** | Effectiveness score 0–100, calculated as Coverage % × Uniformity |

When both images are analysed, the **Eff. Score** cells are colour-coded:
- **Green** — the better-performing image
- **Red** — the lower-performing image

---

## Droplet Size Distribution Chart

A grouped bar chart comparing the two images across three size bins:

| Bin | Diameter range |
|---|---|
| Small | < 20 px |
| Medium | 20 – 49 px |
| Large | ≥ 50 px |

Red bars = No Adjuvant, Green bars = With Adjuvant.

---

## Analysis Pipeline (Technical Summary)

1. **Leaf segmentation** — converts to HSV colour space and thresholds on green hue (0.18–0.55), saturation ≥ 0.10, and value ≥ 0.10. Morphological closing and opening clean up the mask. Only the largest connected component is kept as the leaf region.

2. **Wet-area detection** — converts to LAB colour space. The 75th-percentile lightness of leaf pixels is the "dry leaf" reference. Pixels deviating from this by more than the scaled threshold are classified as spray-contacted, then cleaned with a small morphological open/close.

3. **Blob analysis** — connected components in the contacted mask are labelled. Blobs smaller than 10 px are ignored. Per-blob area and equivalent diameter are recorded.

4. **Metrics** — Coverage % is contacted pixels / leaf pixels. Uniformity uses normalised Shannon entropy over a spatial grid. Effectiveness Score = Coverage % × Uniformity.

5. **Overlay rendering** — background dimmed to 40%, contacted region blended with a green channel, blob contours drawn in red.

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-image` | Colour-space conversion, morphology, connected-component analysis |
| `numpy` | Array operations |
| `matplotlib` | Bar chart and report export |
| `Pillow` | Image loading, overlay drawing |
| `tkinter` | GUI (standard library, no install needed) |
