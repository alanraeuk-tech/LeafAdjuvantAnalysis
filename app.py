"""
app.py — Streamlit interface for the WSP Adjuvant Analyser.

Run locally:
    streamlit run app.py

Deploy: push to GitHub and connect at share.streamlit.io
"""

import csv
import io
import textwrap

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.gridspec import GridSpec
from PIL import Image

from analyser import (DEFAULT_CARD_HEIGHT_MM, DEFAULT_CARD_WIDTH_MM,
                      DEFAULT_MIN_DIAMETER_PX, DEFAULT_THRESHOLD,
                      analyse_image, calibrate_from_card, upscale_for_display)
from metrics import DepositMetrics, cumulative_volume_curve, summarise


APP_VERSION = "2.0"
LOGO_URL = "https://www.desangosse.co.uk/assets/images/desangosse-logo-final.svg"
LABELS = ["No Adjuvant", "With Adjuvant"]
COLOURS = ["#8f9397", "#4cbd89"]   # grey = no adjuvant, teal = with

# (attribute, label, format, "higher is better"? None = no judgement)
ROWS = [
    ("coverage_pct",    "Coverage",                 "{:.1f} %",  True),
    ("density_per_cm2", "Droplet density",          "{:.1f} /cm²", None),
    ("droplet_count",   "Droplets (after split)",   "{:d}",      None),
    ("stain_count",     "Stains (as seen)",         "{:d}",      None),
    ("merged_pct",      "Coverage in merged stains", "{:.0f} %", None),
    ("dv10_um",         "Dv0.1",                    "{:.0f} µm", None),
    ("dv50_um",         "Dv0.5 (VMD)",              "{:.0f} µm", None),
    ("dv90_um",         "Dv0.9",                    "{:.0f} µm", None),
    ("nmd_um",          "NMD",                      "{:.0f} µm", None),
    ("relative_span",   "Relative span",            "{:.2f}",    False),
    ("coverage_cv_pct", "Coverage CV (unevenness)", "{:.0f} %",  False),
]


def fmt(metrics: DepositMetrics | None, attr: str, pattern: str) -> str:
    if metrics is None:
        return "—"
    return pattern.format(getattr(metrics, attr))


@st.cache_data(max_entries=8, show_spinner=False)
def load_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"),
                    dtype=np.uint8)


@st.cache_data(max_entries=8, show_spinner="Analysing card…")
def run_analysis(data: bytes, threshold: float, min_d: float, split: bool):
    return analyse_image(load_image(data), threshold, min_d, split)


def comparison_rows(computed):
    rows = []
    for attr, label, pattern, _ in ROWS:
        row = {"Metric": label}
        for slot, name in enumerate(LABELS):
            row[name] = fmt(computed[slot], attr, pattern)
        if computed[0] and computed[1]:
            a, b = getattr(computed[0], attr), getattr(computed[1], attr)
            if pattern.endswith("%"):
                row["Change"] = f"{b - a:+.0f} pts"
            else:
                row["Change"] = f"{(b - a) / a:+.0%}" if a else "—"
        rows.append(row)
    return rows


def volume_chart(computed, diameters_um, ax):
    for slot, (name, col) in enumerate(zip(LABELS, COLOURS)):
        m = computed[slot]
        if m is None:
            continue
        d, cum = cumulative_volume_curve(diameters_um[slot])
        if d.size:
            ax.step(d, cum, where="post", color=col, label=name, lw=2)
            ax.axvline(m.dv50_um, color=col, ls="--", lw=1)
    ax.axhline(50, color="#dfe2e5", lw=1, zorder=0)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Stain diameter (µm)")
    ax.set_ylabel("Cumulative spray volume (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Volume distribution (dashed = Dv0.5)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


def build_report_png(images, results, computed, diameters_um) -> bytes:
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15,
                  width_ratios=[1, 1, 2.6])
    for slot, name in enumerate(LABELS):
        for row, (img, title) in enumerate([
                (images[slot], "Original"),
                (results[slot].overlay_image if results[slot] else None,
                 "Detected")]):
            ax = fig.add_subplot(gs[row, slot])
            if img is not None:
                ax.imshow(img)
            ax.set_title(f"{name} — {title}", fontsize=9)
            ax.axis("off")

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")
    lines = [f"{'Metric':<26}{'No adj.':>12}{'With adj.':>12}", "─" * 50]
    for attr, label, pattern, _ in ROWS:
        lines.append(f"{label:<26}{fmt(computed[0], attr, pattern):>12}"
                     f"{fmt(computed[1], attr, pattern):>12}")
    ax_txt.text(0, 1, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8, va="top", family="monospace")
    notes = []
    for slot, name in enumerate(LABELS):
        for w in (computed[slot].warnings if computed[slot] else []):
            notes.append(f"{name}: {w}")
    if notes:
        wrapped = "\n".join(textwrap.fill(n, 95) for n in notes)
        ax_txt.text(0, 0.38, wrapped, transform=ax_txt.transAxes,
                    fontsize=7, va="top", color="#a05a00")

    volume_chart(computed, diameters_um, fig.add_subplot(gs[1, 2]))
    fig.suptitle("WSP Adjuvant Analysis Report", fontsize=13,
                 fontweight="bold", color="#0C6951")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_csv(computed) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["metric", "no_adjuvant", "with_adjuvant"])
    for attr, *_ in ROWS + [("um_per_px", None, None, None)]:
        w.writerow([attr] + [round(getattr(m, attr), 3) if m else ""
                             for m in computed])
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="WSP Adjuvant Analyser", layout="wide")
st.markdown("""
<style>
.brand-header { border-top: 4px solid #4cbd89; padding: .75rem 0 .5rem;
  margin-bottom: .5rem; display: flex; align-items: center; gap: 1.5rem; }
.ds-footer { margin-top: 2.5rem; border-top: 1px solid #dfe2e5;
  padding-top: .75rem; font-size: .78rem; color: #8f9397; text-align: center; }
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="brand-header">
  <img src="{LOGO_URL}" height="54" alt="De Sangosse">
  <div>
    <div style="font-size:1.6rem;font-weight:700;color:#343a40;">Adjuvant Spray Analyser</div>
    <div style="font-size:.9rem;color:#8f9397;">Water-Sensitive Paper (WSP) Analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Card size")
    st.caption("Photos must be cropped to the card edges. "
               "Standard cards are 26 × 76 mm.")
    card_w = st.number_input("Width (mm)", 5.0, 300.0,
                             DEFAULT_CARD_WIDTH_MM, 1.0)
    card_h = st.number_input("Height (mm)", 5.0, 300.0,
                             DEFAULT_CARD_HEIGHT_MM, 1.0)

    st.header("Detection")
    threshold = st.slider(
        "Sensitivity threshold", 10, 80, DEFAULT_THRESHOLD,
        help="Colour difference from the paper needed to count as a stain. "
             "Lower picks up fainter stains but also smudges and shadows.")
    min_d = st.slider(
        "Ignore stains smaller than (px)", 1.0, 6.0,
        DEFAULT_MIN_DIAMETER_PX, 0.5,
        help="Noise floor. Check the overlay: raise it if specks of dirt "
             "are being counted.")
    split = st.checkbox(
        "Separate overlapping stains", True,
        help="Splits merged stains at their narrow points so droplets are "
             "counted individually.")

    st.header("Reporting")
    cell_mm = st.slider("Evenness cell size (mm)", 2.0, 13.0, 5.0, 0.5)
    spread = st.number_input(
        "Spread factor", 1.0, 3.0, 1.0, 0.05,
        help="Stains are bigger than the droplets that made them. Leave at "
             "1.0 to report stain sizes; set it if you have a calibrated "
             "factor for your paper and liquid.")

col1, col2 = st.columns(2)
uploads = []
for col, name, key in [(col1, LABELS[0], "u1"), (col2, LABELS[1], "u2")]:
    with col:
        st.subheader(name)
        uploads.append(st.file_uploader(
            name, type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key=key, label_visibility="collapsed"))

images, results, computed = [None, None], [None, None], [None, None]
diameters_um = [None, None]
for slot, upload in enumerate(uploads):
    if upload is None:
        continue
    data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    img = load_image(data)
    cal = calibrate_from_card(img.shape[1], img.shape[0], card_w, card_h)
    res = run_analysis(data, threshold, min_d, split)
    m = summarise(res, cal, cell_mm, spread)
    images[slot], results[slot], computed[slot] = img, res, m
    diameters_um[slot] = res.diameters_px * cal.um_per_px / spread

if not any(results):
    st.info("Upload a scan or photo of a sprayed card to start. "
            "Upload both to compare.")
else:
    cols = st.columns(2)
    for slot, col in enumerate(cols):
        if results[slot] is None:
            continue
        with col:
            show = upscale_for_display(
                np.hstack([results[slot].image_rgb,
                           np.full((results[slot].image_rgb.shape[0], 4, 3),
                                   255, np.uint8),
                           results[slot].overlay_image]))
            st.image(show, caption="Original | Detected (red = droplet edges)",
                     width="stretch")
            for w in computed[slot].warnings:
                st.warning(w)

    st.markdown("---")
    st.subheader("Results")
    st.table(comparison_rows(computed))

    if computed[0] and computed[1]:
        a, b = computed
        st.markdown(
            f"**Coverage** {a.coverage_pct:.0f}% → {b.coverage_pct:.0f}%. "
            f"**Coverage in merged stains** {a.merged_pct:.0f}% → "
            f"{b.merged_pct:.0f}%: a big rise usually means droplets are "
            f"spreading further on contact, which is what most adjuvants "
            f"are for.")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    volume_chart(computed, diameters_um, ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    c1.download_button("Download report (PNG)",
                       build_report_png(images, results, computed, diameters_um),
                       "wsp_adjuvant_report.png", "image/png")
    c2.download_button("Download figures (CSV)", build_csv(computed),
                       "wsp_adjuvant_metrics.csv", "text/csv")

st.markdown(f'<div class="ds-footer">De Sangosse WSP Adjuvant Analysis Tool'
            f' &nbsp;·&nbsp; v{APP_VERSION}</div>', unsafe_allow_html=True)
