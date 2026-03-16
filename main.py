"""
main.py — GUI entry point for the Leaf Adjuvant Analyser.

Layout (3-column tkinter window):
    Col 0  Image 1 panel  (No Adjuvant)   — orig + overlay Canvas, stacked
    Col 1  Image 2 panel  (With Adjuvant) — orig + overlay Canvas, stacked
    Col 2  Results panel                  — metrics table + chart
    Row 1  Bottom bar                     — threshold slider, grid size, buttons

Images are Canvas widgets that fill their cell and rescale (up or down) on
window resize.  Images are stored as RGB uint8 numpy arrays.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analyser import analyse_image, AnalysisResult
from metrics import coverage_percent, uniformity_score, effectiveness_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    ("coverage",   "Coverage %",      lambda v: f"{v:.1f} %"),
    ("droplets",   "Droplets",        lambda v: str(int(v))),
    ("mean_diam",  "Mean diam (px)",  lambda v: f"{v:.1f}"),
    ("uniformity", "Uniformity",      lambda v: f"{v:.3f}"),
    ("score",      "Eff. Score",      lambda v: f"{v:.1f}"),
]

BIN_LABELS = ["Small\n(<20 px)", "Medium\n(20–50 px)", "Large\n(≥50 px)"]
BAR_COLORS = ["#e74c3c", "#2ecc71"]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class LeafAnalyserApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Leaf Adjuvant Analyser")
        self.root.minsize(1100, 700)

        self.images_rgb: list[np.ndarray | None] = [None, None]
        self.results:    list[AnalysisResult | None] = [None, None]
        self._debounce_id: str | None = None

        # Canvas widgets for image display — populated by _build_image_panel
        # orig_canvases[slot], overlay_canvases[slot]
        self.orig_canvases:    list[tk.Canvas] = []
        self.overlay_canvases: list[tk.Canvas] = []

        # {key: (label_slot0, label_slot1)}
        self.metric_labels: dict[str, tuple[tk.Label, tk.Label]] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Two image columns share equally; results panel is narrower
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=3)
        outer.columnconfigure(2, weight=2)
        outer.rowconfigure(0, weight=1)
        outer.rowconfigure(1, weight=0)

        self._build_image_panel(outer, slot=0, col=0,
                                title="Image 1 — No Adjuvant")
        self._build_image_panel(outer, slot=1, col=1,
                                title="Image 2 — With Adjuvant")

        results_frame = tk.LabelFrame(outer, text="Results", padx=6, pady=6)
        results_frame.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        self._build_results_panel(results_frame)

        bottom = tk.Frame(outer, pady=4)
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew")
        self._build_bottom_bar(bottom)

    def _build_image_panel(self, parent: tk.Frame, slot: int, col: int,
                           title: str) -> None:
        frame = tk.LabelFrame(parent, text=title, padx=4, pady=4)
        frame.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)

        frame.columnconfigure(0, weight=1)
        # Button row fixed; the two canvas rows share remaining height equally
        frame.rowconfigure(0, weight=0)
        frame.rowconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

        tk.Button(
            frame, text=f"Load Image {slot + 1}",
            command=lambda s=slot: self.load_image(s),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))

        orig_cv = tk.Canvas(frame, bg="#1a1a1a", highlightthickness=0)
        orig_cv.grid(row=1, column=0, sticky="nsew", pady=(0, 3))

        overlay_cv = tk.Canvas(frame, bg="#1a1a1a", highlightthickness=0)
        overlay_cv.grid(row=2, column=0, sticky="nsew")

        # Draw placeholder text
        self._canvas_placeholder(orig_cv,    "No image loaded")
        self._canvas_placeholder(overlay_cv, "Overlay")

        # Re-render image when canvas is resized (window drag)
        orig_cv.bind(
            "<Configure>",
            lambda e, s=slot: self._on_canvas_resize(e, s, is_overlay=False),
        )
        overlay_cv.bind(
            "<Configure>",
            lambda e, s=slot: self._on_canvas_resize(e, s, is_overlay=True),
        )

        self.orig_canvases.append(orig_cv)
        self.overlay_canvases.append(overlay_cv)

    def _build_results_panel(self, parent: tk.LabelFrame) -> None:
        # Metrics table
        tbl = tk.Frame(parent)
        tbl.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        for c, (hdr, w) in enumerate(
            [("Metric", 15), ("No Adj.", 10), ("With Adj.", 10)]
        ):
            tk.Label(
                tbl, text=hdr, font=("Helvetica", 9, "bold"),
                width=w, anchor="center",
            ).grid(row=0, column=c, padx=2, pady=(0, 2))

        for r, (key, label, _fmt) in enumerate(METRIC_KEYS, start=1):
            tk.Label(tbl, text=label, anchor="w", width=15).grid(
                row=r, column=0, sticky="w", padx=2, pady=1
            )
            lbl1 = tk.Label(tbl, text="—", width=10, anchor="center",
                            relief=tk.GROOVE)
            lbl1.grid(row=r, column=1, padx=2, pady=1)
            lbl2 = tk.Label(tbl, text="—", width=10, anchor="center",
                            relief=tk.GROOVE)
            lbl2.grid(row=r, column=2, padx=2, pady=1)
            self.metric_labels[key] = (lbl1, lbl2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=0, column=0, sticky="ew"
        )

        # Matplotlib chart
        self.fig = Figure(figsize=(3.8, 3.5), dpi=90)
        self.ax  = self.fig.add_subplot(111)
        self._init_chart()

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.chart_canvas.get_tk_widget().grid(
            row=1, column=0, sticky="nsew", pady=(8, 0)
        )

    def _build_bottom_bar(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        self.threshold_var = tk.IntVar(value=30)
        ttk.Scale(
            parent, from_=5, to=100, orient=tk.HORIZONTAL,
            variable=self.threshold_var, length=200,
        ).pack(side=tk.LEFT)
        self.threshold_display = tk.Label(parent, text="30", width=3,
                                          font=("Courier", 9, "bold"))
        self.threshold_display.pack(side=tk.LEFT, padx=(2, 16))
        self.threshold_var.trace_add("write", self._on_param_change)

        tk.Label(parent, text="Grid size:").pack(side=tk.LEFT, padx=(0, 4))
        self.grid_var = tk.StringVar(value="8")
        ttk.Spinbox(
            parent, from_=4, to=16, increment=2,
            textvariable=self.grid_var, width=4,
        ).pack(side=tk.LEFT, padx=(0, 16))
        self.grid_var.trace_add("write", self._on_param_change)

        ttk.Button(parent, text="Analyse",     command=self.analyse
                   ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(parent, text="Save Report", command=self.save_report
                   ).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------

    def _canvas_placeholder(self, canvas: tk.Canvas, text: str) -> None:
        """Draw centered placeholder text (called before any image is loaded)."""
        canvas.delete("all")
        canvas.update_idletasks()
        w = max(canvas.winfo_width(), 10)
        h = max(canvas.winfo_height(), 10)
        canvas.create_text(w // 2, h // 2, text=text,
                           fill="#555", font=("Helvetica", 10), anchor="center",
                           tags="placeholder")

    def _draw_image_on_canvas(self, img_rgb: np.ndarray,
                               canvas: tk.Canvas) -> None:
        """Scale img_rgb to fill the canvas (up OR down) and draw it centred."""
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        pil = Image.fromarray(img_rgb)
        src_w, src_h = pil.size

        # Scale to fill canvas while preserving aspect ratio
        scale = min(cw / src_w, ch / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")
        canvas.image = tk_img   # type: ignore[attr-defined] — prevent GC

    def _on_canvas_resize(self, event: tk.Event,
                           slot: int, is_overlay: bool) -> None:
        """Re-render the image when the canvas is resized (e.g. window drag)."""
        if is_overlay:
            r = self.results[slot]
            img = r.overlay_image if r is not None else None
            canvas = self.overlay_canvases[slot]
        else:
            img = self.images_rgb[slot]
            canvas = self.orig_canvases[slot]

        if img is not None:
            self._draw_image_on_canvas(img, canvas)

    # ------------------------------------------------------------------
    # Parameter change (slider / spinbox)
    # ------------------------------------------------------------------

    def _on_param_change(self, *_args) -> None:
        try:
            val = int(self.threshold_var.get())
            self.threshold_display.config(text=str(val))
        except (tk.TclError, ValueError):
            pass
        if self._debounce_id:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(600, self._auto_analyse)

    def _auto_analyse(self) -> None:
        if self.images_rgb[0] is not None and self.images_rgb[1] is not None:
            self.analyse()

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------

    def load_image(self, slot: int) -> None:
        path = filedialog.askopenfilename(
            title=f"Load Image {slot + 1}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files",   "*.*"),
            ],
        )
        if not path:
            return

        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read image:\n{exc}")
            return

        self.images_rgb[slot] = np.array(pil_img, dtype=np.uint8)
        self.results[slot]    = None

        self._draw_image_on_canvas(self.images_rgb[slot],
                                   self.orig_canvases[slot])
        self._canvas_placeholder(self.overlay_canvases[slot],
                                 "Not analysed yet")

    # ------------------------------------------------------------------
    # Analyse
    # ------------------------------------------------------------------

    def analyse(self) -> None:
        if self.images_rgb[0] is None and self.images_rgb[1] is None:
            messagebox.showwarning("No Images", "Load at least one image first.")
            return

        try:
            threshold = int(self.threshold_var.get())
        except (tk.TclError, ValueError):
            threshold = 30

        try:
            grid_n = max(2, min(int(self.grid_var.get()), 20))
        except (tk.TclError, ValueError):
            grid_n = 8

        for slot in range(2):
            if self.images_rgb[slot] is None:
                self.results[slot] = None
                continue
            result = analyse_image(self.images_rgb[slot], threshold=threshold)
            self.results[slot] = result
            self._draw_image_on_canvas(result.overlay_image,
                                       self.overlay_canvases[slot])

        self._update_results_panel(grid_n)

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _update_results_panel(self, grid_n: int) -> None:
        computed: list[dict | None] = []
        for slot in range(2):
            r = self.results[slot]
            if r is None:
                computed.append(None)
                continue
            cov = coverage_percent(r.contacted_mask, r.leaf_mask)
            uni = uniformity_score(r.contacted_mask, r.leaf_mask, grid_n)
            eff = effectiveness_score(cov, uni)
            computed.append({
                "coverage":   cov,
                "droplets":   r.droplet_count,
                "mean_diam":  r.mean_diameter,
                "uniformity": uni,
                "score":      eff,
                "bins":       r.size_bins,
            })

        for key, _label, fmt in METRIC_KEYS:
            lbl0, lbl1 = self.metric_labels[key]
            v0 = fmt(computed[0][key]) if computed[0] else "—"
            v1 = fmt(computed[1][key]) if computed[1] else "—"
            lbl0.config(text=v0, fg="black")
            lbl1.config(text=v1, fg="black")

            if key == "score" and computed[0] and computed[1]:
                if computed[1]["score"] > computed[0]["score"]:
                    lbl0.config(fg="#c0392b")
                    lbl1.config(fg="#27ae60")
                elif computed[0]["score"] > computed[1]["score"]:
                    lbl0.config(fg="#27ae60")
                    lbl1.config(fg="#c0392b")

        self.ax.clear()
        self._init_chart()
        x = np.arange(3)
        w = 0.35
        for slot, (name, color) in enumerate(zip(["No Adj.", "With Adj."],
                                                  BAR_COLORS)):
            if computed[slot]:
                self.ax.bar(
                    x + (slot - 0.5) * w,
                    computed[slot]["bins"],
                    w, label=name, color=color, alpha=0.85,
                )
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(BIN_LABELS, fontsize=7)
        self.ax.legend(fontsize=7)
        self.fig.tight_layout(pad=0.8)
        self.chart_canvas.draw()

    def _init_chart(self) -> None:
        self.ax.set_title("Droplet Size Distribution", fontsize=8)
        self.ax.set_ylabel("Count", fontsize=7)
        self.ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    def save_report(self) -> None:
        if not any(self.results):
            messagebox.showwarning("No Results", "Run analysis before saving.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")],
            title="Save Report As",
        )
        if not path:
            return
        try:
            self._export_report(path)
            messagebox.showinfo("Saved", f"Report saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _export_report(self, path: str) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        try:
            grid_n = max(2, min(int(self.grid_var.get()), 20))
        except (tk.TclError, ValueError):
            grid_n = 8

        fig = plt.figure(figsize=(15, 8))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)
        panel_labels = ["No Adjuvant", "With Adjuvant"]

        for slot in range(2):
            ax_orig = fig.add_subplot(gs[0, slot])
            if self.images_rgb[slot] is not None:
                ax_orig.imshow(self.images_rgb[slot])
            ax_orig.set_title(f"{panel_labels[slot]} — Original", fontsize=9)
            ax_orig.axis("off")

            ax_ov = fig.add_subplot(gs[1, slot])
            if self.results[slot] is not None:
                ax_ov.imshow(self.results[slot].overlay_image)
            ax_ov.set_title(f"{panel_labels[slot]} — Overlay", fontsize=9)
            ax_ov.axis("off")

        ax_txt = fig.add_subplot(gs[0, 2])
        ax_txt.axis("off")

        computed: list[dict | None] = []
        for slot in range(2):
            r = self.results[slot]
            if r is None:
                computed.append(None)
            else:
                cov = coverage_percent(r.contacted_mask, r.leaf_mask)
                uni = uniformity_score(r.contacted_mask, r.leaf_mask, grid_n)
                eff = effectiveness_score(cov, uni)
                computed.append({
                    "coverage":   cov,
                    "droplets":   r.droplet_count,
                    "mean_diam":  r.mean_diameter,
                    "uniformity": uni,
                    "score":      eff,
                })

        lines = ["Metrics Comparison", ""]
        lines.append(f"{'Metric':<18}  {'No Adj.':>9}  {'With Adj.':>9}")
        lines.append("─" * 42)
        for key, label, fmt in METRIC_KEYS:
            v0 = fmt(computed[0][key]) if computed[0] else "—"
            v1 = fmt(computed[1][key]) if computed[1] else "—"
            lines.append(f"{label:<18}  {v0:>9}  {v1:>9}")
        ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                    fontsize=8.5, verticalalignment="top",
                    fontfamily="monospace")

        ax_chart = fig.add_subplot(gs[1, 2])
        x = np.arange(3)
        w = 0.35
        for slot, (name, color) in enumerate(zip(panel_labels, BAR_COLORS)):
            if self.results[slot] is not None:
                ax_chart.bar(x + (slot - 0.5) * w,
                             self.results[slot].size_bins,
                             w, label=name, color=color, alpha=0.85)
        ax_chart.set_xticks(x)
        ax_chart.set_xticklabels(["Small", "Medium", "Large"], fontsize=8)
        ax_chart.set_title("Droplet Size Distribution", fontsize=9)
        ax_chart.legend(fontsize=8)

        fig.suptitle("Leaf Adjuvant Effectiveness Analysis Report",
                     fontsize=13, fontweight="bold")

        if path.lower().endswith(".pdf"):
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(path) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
        else:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = LeafAnalyserApp(root)
    root.mainloop()
