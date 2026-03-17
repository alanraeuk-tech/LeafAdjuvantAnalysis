"""
app.py — Streamlit web interface for the WSP Adjuvant Analyser.

Run locally:
    streamlit run app.py

Deploy: push to GitHub and connect at share.streamlit.io
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.gridspec import GridSpec
from PIL import Image

from analyser import analyse_image, AnalysisResult
from metrics import coverage_percent, uniformity_score, effectiveness_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    ("coverage",   "Coverage %",     lambda v: f"{v:.1f} %"),
    ("droplets",   "Droplets",       lambda v: str(int(v))),
    ("mean_diam",  "Mean diam (px)", lambda v: f"{v:.1f}"),
    ("uniformity", "Uniformity",     lambda v: f"{v:.3f}"),
    ("score",      "Eff. Score",     lambda v: f"{v:.1f}"),
]

BIN_LABELS = ["Small (<20 px)", "Medium (20–50 px)", "Large (≥50 px)"]
BAR_COLORS = ["#e74c3c", "#2ecc71"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(uploaded_file) -> np.ndarray:
    return np.array(Image.open(uploaded_file).convert("RGB"), dtype=np.uint8)


def compute_metrics(result: AnalysisResult, grid_n: int) -> dict:
    cov = coverage_percent(result.contacted_mask, result.leaf_mask)
    uni = uniformity_score(result.contacted_mask, result.leaf_mask, grid_n)
    eff = effectiveness_score(cov, uni)
    return {
        "coverage":   cov,
        "droplets":   result.droplet_count,
        "mean_diam":  result.mean_diameter,
        "uniformity": uni,
        "score":      eff,
        "bins":       result.size_bins,
    }


def build_report_png(images_rgb, results, computed, grid_n: int) -> bytes:
    fig = plt.figure(figsize=(15, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)
    panel_labels = ["No Adjuvant", "With Adjuvant"]

    for slot in range(2):
        ax_orig = fig.add_subplot(gs[0, slot])
        if images_rgb[slot] is not None:
            ax_orig.imshow(images_rgb[slot])
        ax_orig.set_title(f"{panel_labels[slot]} — Original", fontsize=9)
        ax_orig.axis("off")

        ax_ov = fig.add_subplot(gs[1, slot])
        if results[slot] is not None:
            ax_ov.imshow(results[slot].overlay_image)
        ax_ov.set_title(f"{panel_labels[slot]} — Overlay", fontsize=9)
        ax_ov.axis("off")

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")
    lines = ["Metrics Comparison", ""]
    lines.append(f"{'Metric':<18}  {'No Adj.':>9}  {'With Adj.':>9}")
    lines.append("─" * 42)
    for key, label, fmt in METRIC_KEYS:
        v0 = fmt(computed[0][key]) if computed[0] else "—"
        v1 = fmt(computed[1][key]) if computed[1] else "—"
        lines.append(f"{label:<18}  {v0:>9}  {v1:>9}")
    ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8.5, verticalalignment="top", fontfamily="monospace")

    ax_chart = fig.add_subplot(gs[1, 2])
    x = np.arange(3)
    w = 0.35
    for slot, (name, color) in enumerate(zip(panel_labels, BAR_COLORS)):
        if results[slot] is not None:
            ax_chart.bar(x + (slot - 0.5) * w, results[slot].size_bins,
                         w, label=name, color=color, alpha=0.85)
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(["Small", "Medium", "Large"], fontsize=8)
    ax_chart.set_title("Droplet Size Distribution", fontsize=9)
    ax_chart.legend(fontsize=8)

    fig.suptitle("WSP Adjuvant Effectiveness Analysis Report",
                 fontsize=13, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="WSP Adjuvant Analyser", layout="wide")
st.title("Adjuvant Spray Analyser — Water-Sensitive Paper")

# --- Sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    threshold = st.slider("Threshold", min_value=5, max_value=100, value=30)
    grid_n    = st.number_input("Grid size", min_value=4, max_value=16,
                                value=8, step=2)
    st.markdown("---")
    st.caption("Threshold controls minimum colour saturation for blue-droplet detection. "
               "Grid size sets uniformity resolution.")

# --- Image upload -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Paper 1 — No Adjuvant")
    upload1 = st.file_uploader("Upload image 1",
                               type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
                               key="upload1", label_visibility="collapsed")
with col2:
    st.subheader("Paper 2 — With Adjuvant")
    upload2 = st.file_uploader("Upload image 2",
                               type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
                               key="upload2", label_visibility="collapsed")

# --- Load & analyse ---------------------------------------------------------
images_rgb = [None, None]
results    = [None, None]

for slot, upload in enumerate([upload1, upload2]):
    if upload is not None:
        images_rgb[slot] = load_image(upload)
        results[slot]    = analyse_image(images_rgb[slot], threshold=threshold)

# --- Display images ---------------------------------------------------------
if any(r is not None for r in results):
    st.markdown("---")
    img_col1, img_col2 = st.columns(2)
    panels = [img_col1, img_col2]
    labels = ["No Adjuvant", "With Adjuvant"]

    for slot, (panel, label) in enumerate(zip(panels, labels)):
        with panel:
            if images_rgb[slot] is not None:
                st.image(images_rgb[slot], caption=f"{label} — Original",
                         use_container_width=True)
                st.image(results[slot].overlay_image,
                         caption=f"{label} — Overlay",
                         use_container_width=True)

    # --- Metrics table ------------------------------------------------------
    st.markdown("---")
    st.subheader("Results")

    computed = [None, None]
    for slot in range(2):
        if results[slot] is not None:
            computed[slot] = compute_metrics(results[slot], int(grid_n))

    table_data = {"Metric": [], "No Adjuvant": [], "With Adjuvant": []}
    for key, label, fmt in METRIC_KEYS:
        table_data["Metric"].append(label)
        table_data["No Adjuvant"].append(
            fmt(computed[0][key]) if computed[0] else "—"
        )
        table_data["With Adjuvant"].append(
            fmt(computed[1][key]) if computed[1] else "—"
        )
    st.table(table_data)

    # Highlight which image scored better
    if computed[0] and computed[1]:
        s0, s1 = computed[0]["score"], computed[1]["score"]
        if s1 > s0:
            st.success(f"With Adjuvant scores higher ({s1:.1f} vs {s0:.1f})")
        elif s0 > s1:
            st.warning(f"No Adjuvant scores higher ({s0:.1f} vs {s1:.1f})")
        else:
            st.info("Both images score equally.")

    # --- Droplet size chart -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(3)
    w = 0.35
    for slot, (name, color) in enumerate(zip(labels, BAR_COLORS)):
        if computed[slot]:
            ax.bar(x + (slot - 0.5) * w, computed[slot]["bins"],
                   w, label=name, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_title("Droplet Size Distribution")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Download report ----------------------------------------------------
    st.markdown("---")
    report_bytes = build_report_png(images_rgb, results, computed, int(grid_n))
    st.download_button(
        label="Download Report (PNG)",
        data=report_bytes,
        file_name="wsp_adjuvant_report.png",
        mime="image/png",
    )

else:
    st.info("Upload at least one WSP image above to begin analysis.")
