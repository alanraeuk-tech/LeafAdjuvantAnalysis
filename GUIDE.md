# WSP Adjuvant Analyser: User Guide

## What it does

Compares how a spray lands with and without an adjuvant, using water-sensitive paper (WSP). WSP is yellow and turns blue wherever a droplet hits it. You upload a photo or scan of a card sprayed without the adjuvant and one sprayed with it, and the tool measures the stains on each and puts the figures side by side.

```bash
streamlit run app.py
```

---

## Getting good images

This matters more than any setting in the app.

- **Scan rather than photograph if you can**, at **600 dpi or better** (1200 dpi is ideal). A phone photo of a whole card is usually around 100 µm per pixel, which can't see fine droplets at all. The app warns you when resolution is too low.
- **Crop to the card edges.** The app works out the scale from the card size, so any background left in the image throws the micron figures off. The app warns if the image shape doesn't match the card.
- Even lighting, no shadows, no glare. Handle cards by the edges with dry gloves.
- Spray at a rate that gives under about 30% coverage if you want droplet sizes. Above that, stains overlap so much that sizing becomes guesswork. Coverage itself stays reliable.

---

## Settings (sidebar)

### Card size
Width and height of the card in mm. Standard cards are 26 × 76 mm. If you cut cards down, enter the cut size.

### Sensitivity threshold (default 35)
How different from the paper colour a pixel must be to count as stained, measured as colour distance (ΔE). Lower picks up fainter stains but also smudges and shadows; higher misses faint fine droplets. Check the overlay: red outlines should hug the visible stains.

### Ignore stains smaller than (default 2 px)
Noise floor. Raise it if dust or paper texture is being counted.

### Separate overlapping stains (default on)
Where two droplets landed on top of each other, splits the merged stain at its narrow point so each is counted. Turn off to count stains exactly as they appear.

### Evenness cell size (default 5 mm)
Grid size used for the coverage CV figure.

### Spread factor (default 1.0)
A droplet spreads when it hits the paper, so the stain is bigger than the droplet. At 1.0 the app reports stain sizes. If you have a calibrated spread factor for your paper and liquid, enter it and sizes become droplet sizes (stain ÷ factor). Note an adjuvant usually changes the spread factor, which is part of what you're measuring.

---

## The overlay

Each card shows the original next to the detected version:

| You see | Meaning |
|---|---|
| Green tint | Area counted as stained |
| Red outline | Edge of each droplet, including split lines between merged ones |
| Plain yellow | Paper with no stain detected |

---

## Results

| Metric | What it means |
|---|---|
| **Coverage** | % of the card that's stained. The most reliable figure. |
| **Droplet density** | Droplets per cm². Label recommendations are often given in this form. |
| **Droplets (after split)** | Count after separating overlaps. |
| **Stains (as seen)** | Count of separate blue patches before splitting. |
| **Coverage in merged stains** | Share of the stained area in patches that contained more than one droplet. A big rise with adjuvant usually means droplets are spreading further on contact. |
| **Dv0.1 / Dv0.5 / Dv0.9** | 10%, 50% and 90% of the spray *volume* is in stains smaller than this. Dv0.5 is the volume median diameter (VMD), the standard headline droplet-size figure. |
| **NMD** | Number median diameter: half the droplets are smaller than this. Much lower than VMD because lots of small droplets carry little volume. |
| **Relative span** | (Dv0.9 − Dv0.1) / Dv0.5. Lower means more uniform droplet sizes. |
| **Coverage CV** | How unevenly coverage is spread across the card. Lower is more even. |

Where both cards are loaded, a **Change** column shows the difference. Percentage metrics show the change in points (20% → 44% is +24 pts), everything else as a relative change.

The chart plots cumulative spray volume against stain size for both cards. The further right a curve sits, the bigger the stains.

### Warnings

Yellow warnings under each image tell you when the figures shouldn't be trusted: low resolution, image not cropped to the card, or very high coverage. They're also printed on the downloaded report.

---

## Downloads

- **Report (PNG):** both cards, overlays, metrics table, chart and any warnings on one page.
- **Figures (CSV):** the raw numbers, for spreadsheets.

---

## How the analysis works

1. **Trim** a 3 px frame off the image to remove scanner and crop edges.
2. **Find the paper colour:** the median colour of the clearly yellow pixels.
3. **Find stains** in two passes. Globally, any pixel more than the threshold ΔE from the paper colour. Locally, small isolated specks that are faint overall but clearly darker than the paper right around them (fine droplets on lower-resolution images). The local pass only adds small isolated specks, so it can't inflate big stains.
4. **Separate merged stains** with a watershed on the distance transform, so overlapping round stains are cut at their waist. Seeds that are too close to be separate droplets are merged so irregular single stains aren't shredded.
5. **Measure** each droplet's equivalent circular diameter and convert to µm using the card size.
6. **Metrics:** volume percentiles use diameter³ as volume, so a few big stains dominate Dv0.5.
