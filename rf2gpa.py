"""
Ruby fluorescence (DAC pressure) analyzer with multi-Voigt (lmfit) fitting + Tkinter GUI.

Modernized from the legacy Win7/Python3.7 script. Core math/fit flow preserved:
- Read 2-column spectra
- Detect peaks (scipy.signal.find_peaks)
- Fit constant background + N Voigt peaks (lmfit)
- Convert peak center wavelength -> pressure (GPa) + propagated uncertainty

New:
- Modern Python 3.12 compatible
- Multi-file selection + folder import
- Batch analysis with progress + results table
- Overlay compare spectra (raw and fitted)
- Export summary CSV + save per-file plots (optional)
- Optional parallel batch processing (ProcessPoolExecutor)

Dependencies:
    pip install numpy scipy pandas matplotlib lmfit

Run:
    python rf2gpa.py

CLI batch (no GUI):
    python ruby_rf_voigt_gui_py312.py --cli --input "C:\\data" --glob "*.txt" --out "C:\\out" --workers 4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# GUI + plotting
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scipy.signal import find_peaks  # noqa: E402
from lmfit.models import VoigtModel, ConstantModel  # noqa: E402


# -----------------------------
# Core: I/O, fitting, pressure
# -----------------------------

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class AnalysisConfig:
    roi_min_nm: float = 685.0
    roi_max_nm: float = 710.0
    n_peaks: int = 2
    peak_distance: int = 3
    peak_prominence: float = 0.3

    lambda_ref_nm: float = 693.88947  # keep legacy default
    lambda_ref_err_nm: float = 0.05978  # keep legacy default (used in error)
    # If your lab uses different ruby reference (e.g., 694.26 nm or 693.1 nm), set it here in GUI.

    # Fit constraints
    sigma_min: float = 0.05
    sigma_max: float = 2.5
    gamma_min: float = 0.05
    gamma_max: float = 2.5
    max_nfev: int = 3000

    # Which fitted peak to use for the "primary" pressure summary:
    # - "max_amplitude": peak with largest fitted amplitude
    # - "nearest_ref": peak center closest to lambda_ref_nm
    primary_peak_mode: str = "max_amplitude"

    # Plotting compare
    normalize_overlay: bool = True


@dataclass(frozen=True)
class PeakResult:
    peak_index: int
    center_nm: float
    center_err_nm: float
    amplitude: float
    sigma: float
    gamma: float
    pressure_gpa: float
    pressure_err_gpa: float


@dataclass(frozen=True)
class FileResult:
    filepath: str
    ok: bool
    message: str
    peaks: List[PeakResult]
    primary_pressure_gpa: Optional[float] = None
    primary_pressure_err_gpa: Optional[float] = None
    redchi: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None


def _parse_two_column_numeric_lines(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly parse *most* two-column numeric spectra files.
    Ignores lines with alphabetic characters (except 'nm'), and skips obvious non-spectral rows.

    This is more tolerant than a fixed `skiprows=3` approach, and is safer when instruments append
    metadata at the end of the file.
    """
    xs: List[float] = []
    ys: List[float] = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # allow "nm" units but reject other alphabetic content
            s_clean = s.replace("nm", "").replace("NM", "")
            if re.search(r"[A-Za-z]", s_clean):
                continue

            nums = _FLOAT_RE.findall(s_clean)
            if len(nums) < 2:
                continue

            try:
                x = float(nums[0])
                y = float(nums[1])
            except ValueError:
                continue

            # Heuristics to avoid timestamps / headers accidentally parsed as numeric:
            # Ruby spectra are typically within a few hundred to ~1000 nm.
            if not (200.0 <= x <= 2000.0):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            xs.append(x)
            ys.append(y)

    if len(xs) < 20:
        raise ValueError(f"Not enough numeric spectral points parsed from: {path}")

    xarr = np.asarray(xs, dtype=float)
    yarr = np.asarray(ys, dtype=float)

    # Ensure sorted by wavelength
    if np.any(np.diff(xarr) < 0):
        idx = np.argsort(xarr)
        xarr = xarr[idx]
        yarr = yarr[idx]

    return xarr, yarr


def rf_peak_to_gpa(lambda_pk_nm: float, lambda_ref_nm: float) -> float:
    """
    Legacy polynomial conversion preserved from the original tool.
    """
    delta = lambda_pk_nm - lambda_ref_nm
    p = (1870.0 * (delta / lambda_ref_nm)) * (1.0 + 5.63 * (delta / lambda_ref_nm))
    return float(p)


def rf_gpa_error(
    lambda_pk_nm: float,
    lambda_pk_err_nm: Optional[float],
    lambda_ref_nm: float,
    lambda_ref_err_nm: float,
) -> float:
    """
    Conservative uncertainty propagation with guards for near-zero delta.
    """
    pk_err = float(lambda_pk_err_nm) if (lambda_pk_err_nm and lambda_pk_err_nm > 0) else 0.01
    ref_err = float(lambda_ref_err_nm) if (lambda_ref_err_nm and lambda_ref_err_nm > 0) else 0.01

    delta = lambda_pk_nm - lambda_ref_nm
    delta_err = pk_err + ref_err

    # Avoid blowups if delta ~ 0
    if abs(delta) < 1e-9:
        rel_err = (delta_err / lambda_ref_nm)
    else:
        rel_err = (abs(delta) / lambda_ref_nm) * math.sqrt(
            (delta_err / delta) ** 2 + (ref_err / lambda_ref_nm) ** 2
        )
    return float(1870.0 * rel_err)


def detect_top_n_peaks(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int,
    distance: int,
    prominence: float,
) -> List[Dict[str, float]]:
    """
    Find the top-n most prominent peaks in (x, y) and return initial guesses for Voigt fit.
    """
    peaks, props = find_peaks(y, distance=distance, prominence=prominence)
    if peaks.size == 0:
        return []

    prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
    idx = np.argsort(prominences)[-n_peaks:][::-1]
    top = peaks[idx]

    guesses: List[Dict[str, float]] = []
    for p in top:
        guesses.append(
            {
                "center": float(x[p]),
                "amplitude": float(max(y[p], 1.0)),
                "sigma": 0.6,
                "gamma": 0.6,
            }
        )

    # Sort guesses by center for stable numbering
    guesses.sort(key=lambda d: d["center"])
    return guesses


def fit_multi_voigt_lmfit(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    peak_guesses: Sequence[Dict[str, float]],
    cfg: AnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray, "lmfit.model.ModelResult", List[str]]:
    """
    Fit constant background + sum of Voigt peaks to Region of Interest.
    Returns dense x/y for plotting, lmfit result, and list of Voigt component prefixes.
    """
    mask = (x >= xlim[0]) & (x <= xlim[1])
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 10:
        raise ValueError("Region of Interest contains too few points to fit.")

    model = ConstantModel(prefix="c_")
    params = model.make_params(c_c=float(np.median(y_fit)))

    prefixes: List[str] = []
    for i, guess in enumerate(peak_guesses):
        prefix = f"v{i}_"
        prefixes.append(prefix)
        vmod = VoigtModel(prefix=prefix)
        model += vmod
        params.update(vmod.make_params())

        params[f"{prefix}center"].set(value=guess["center"])
        params[f"{prefix}amplitude"].set(value=guess["amplitude"], min=0.0)
        params[f"{prefix}sigma"].set(value=guess.get("sigma", 0.6), min=cfg.sigma_min, max=cfg.sigma_max)
        params[f"{prefix}gamma"].set(value=guess.get("gamma", 0.6), min=cfg.gamma_min, max=cfg.gamma_max)

    result = model.fit(y_fit, params, x=x_fit, max_nfev=cfg.max_nfev)

    x_dense = np.linspace(xlim[0], xlim[1], 600)
    y_dense = model.eval(result.params, x=x_dense)

    return x_dense, y_dense, result, prefixes


def analyze_one_file(path_str: str, cfg: AnalysisConfig) -> FileResult:
    """
    Top-level, picklable worker for batch analysis.
    """
    path = Path(path_str)
    try:
        x, y = _parse_two_column_numeric_lines(path)

        roi = (cfg.roi_min_nm, cfg.roi_max_nm)
        mask = (x >= roi[0]) & (x <= roi[1])
        x_roi = x[mask]
        y_roi = y[mask]
        if x_roi.size < 20:
            raise ValueError(f"Not enough points inside Region of Interest {roi}.")

        guesses = detect_top_n_peaks(
            x_roi,
            y_roi,
            n_peaks=cfg.n_peaks,
            distance=cfg.peak_distance,
            prominence=cfg.peak_prominence,
        )
        if not guesses:
            raise ValueError("No peaks detected; try lowering prominence or widening Region of Interest.")

        x_fit, y_fit, result, prefixes = fit_multi_voigt_lmfit(x, y, roi, guesses, cfg)

        peaks: List[PeakResult] = []
        for i, pfx in enumerate(prefixes):
            center = float(result.params[pfx + "center"].value)
            stderr = result.params[pfx + "center"].stderr
            center_err = float(stderr) if (stderr and stderr > 0) else 0.01

            amp = float(result.params[pfx + "amplitude"].value)
            sigma = float(result.params[pfx + "sigma"].value)
            gamma = float(result.params[pfx + "gamma"].value)

            p_gpa = rf_peak_to_gpa(center, cfg.lambda_ref_nm)
            p_err = rf_gpa_error(center, center_err, cfg.lambda_ref_nm, cfg.lambda_ref_err_nm)

            peaks.append(
                PeakResult(
                    peak_index=i + 1,
                    center_nm=center,
                    center_err_nm=center_err,
                    amplitude=amp,
                    sigma=sigma,
                    gamma=gamma,
                    pressure_gpa=p_gpa,
                    pressure_err_gpa=p_err,
                )
            )

        primary = _select_primary_peak(peaks, cfg)

        return FileResult(
            filepath=str(path),
            ok=True,
            message="OK",
            peaks=peaks,
            primary_pressure_gpa=primary.pressure_gpa if primary else None,
            primary_pressure_err_gpa=primary.pressure_err_gpa if primary else None,
            redchi=float(getattr(result, "redchi", np.nan)),
            aic=float(getattr(result, "aic", np.nan)),
            bic=float(getattr(result, "bic", np.nan)),
        )

    except Exception as e:
        return FileResult(filepath=str(path), ok=False, message=str(e), peaks=[])


def _select_primary_peak(peaks: Sequence[PeakResult], cfg: AnalysisConfig) -> Optional[PeakResult]:
    if not peaks:
        return None
    mode = (cfg.primary_peak_mode or "").strip().lower()
    if mode == "nearest_ref":
        return min(peaks, key=lambda p: abs(p.center_nm - cfg.lambda_ref_nm))
    # default: max amplitude
    return max(peaks, key=lambda p: p.amplitude)



def make_run_stamp() -> str:
    """
    Return a filesystem-safe timestamp for one analysis run.
    Example: 20260518_143022
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamped_output_path(out_dir: Path, filename: str, run_stamp: str) -> Path:
    """
    Build a timestamped output path and avoid overwriting if a file already exists.
    Example:
        ruby_rf_results.csv -> ruby_rf_results_20260518_143022.csv
    If the timestamped file already exists, appends _01, _02, etc.
    """
    base = Path(filename)
    candidate = out_dir / f"{base.stem}_{run_stamp}{base.suffix}"
    counter = 1

    while candidate.exists():
        candidate = out_dir / f"{base.stem}_{run_stamp}_{counter:02d}{base.suffix}"
        counter += 1

    return candidate

def export_results_csv(results: Sequence[FileResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filepath",
                "ok",
                "message",
                "primary_pressure_gpa",
                "primary_pressure_err_gpa",
                "redchi",
                "aic",
                "bic",
                "peak_index",
                "center_nm",
                "center_err_nm",
                "amplitude",
                "sigma",
                "gamma",
                "pressure_gpa",
                "pressure_err_gpa",
            ]
        )
        for r in results:
            if not r.peaks:
                w.writerow(
                    [
                        r.filepath,
                        r.ok,
                        r.message,
                        r.primary_pressure_gpa,
                        r.primary_pressure_err_gpa,
                        r.redchi,
                        r.aic,
                        r.bic,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            else:
                for p in r.peaks:
                    w.writerow(
                        [
                            r.filepath,
                            r.ok,
                            r.message,
                            r.primary_pressure_gpa,
                            r.primary_pressure_err_gpa,
                            r.redchi,
                            r.aic,
                            r.bic,
                            p.peak_index,
                            p.center_nm,
                            p.center_err_nm,
                            p.amplitude,
                            p.sigma,
                            p.gamma,
                            p.pressure_gpa,
                            p.pressure_err_gpa,
                        ]
                    )


# -----------------------------
# GUI
# -----------------------------

class RubyRFApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ruby Fluorescence Voigt Fit (Python 3.12)")
        self.geometry("1200x780")
        self.minsize(1100, 700)

        self._files: List[str] = []
        self._spectra_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._results: Dict[str, FileResult] = {}
        self._color_map: Dict[str, str] = {}  # filepath -> hex color (e.g. '#1f77b4')

        self.cfg = AnalysisConfig()

        self._build_ui()

    # ---- UI layout ----

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        # Left: file list + controls
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="Files").grid(row=0, column=0, sticky="w")

        self.file_list = tk.Listbox(left, selectmode=tk.EXTENDED, height=20)
        self.file_list.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(4, 6))
        yscroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=yscroll.set)
        yscroll.grid(row=1, column=3, sticky="ns", pady=(4, 6))

        btn_row = ttk.Frame(left)
        btn_row.grid(row=2, column=0, columnspan=4, sticky="ew")
        btn_row.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(btn_row, text="Add Files...", command=self._add_files).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(btn_row, text="Add Folder...", command=self._add_folder).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(btn_row, text="Remove", command=self._remove_selected).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        ttk.Button(btn_row, text="Clear", command=self._clear_files).grid(row=0, column=3, padx=2, pady=2, sticky="ew")

        # Options
        opts = ttk.LabelFrame(left, text="Analysis Settings", padding=8)
        opts.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        def add_labeled(row: int, label: str, widget: tk.Widget) -> None:
            ttk.Label(opts, text=label).grid(row=row, column=0, sticky="w", pady=2)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            opts.columnconfigure(1, weight=1)

        self.var_roi_min = tk.StringVar(value=str(self.cfg.roi_min_nm))
        self.var_roi_max = tk.StringVar(value=str(self.cfg.roi_max_nm))
        self.var_n_peaks = tk.StringVar(value=str(self.cfg.n_peaks))
        self.var_prom = tk.StringVar(value=str(self.cfg.peak_prominence))
        self.var_dist = tk.StringVar(value=str(self.cfg.peak_distance))
        self.var_lambda_ref = tk.StringVar(value=str(self.cfg.lambda_ref_nm))
        self.var_lambda_ref_err = tk.StringVar(value=str(self.cfg.lambda_ref_err_nm))
        self.var_workers = tk.StringVar(value="0")
        self.var_norm = tk.BooleanVar(value=self.cfg.normalize_overlay)

        add_labeled(0, "ROI min (nm)", ttk.Entry(opts, textvariable=self.var_roi_min, width=10))
        add_labeled(1, "ROI max (nm)", ttk.Entry(opts, textvariable=self.var_roi_max, width=10))
        add_labeled(2, "Number of peaks", ttk.Spinbox(opts, from_=1, to=5, textvariable=self.var_n_peaks, width=5))
        add_labeled(3, "Peak prominence", ttk.Entry(opts, textvariable=self.var_prom, width=10))
        add_labeled(4, "Peak distance (pts)", ttk.Entry(opts, textvariable=self.var_dist, width=10))
        add_labeled(5, "λ_ref (nm)", ttk.Entry(opts, textvariable=self.var_lambda_ref, width=12))
        add_labeled(6, "λ_ref err (nm)", ttk.Entry(opts, textvariable=self.var_lambda_ref_err, width=12))
        add_labeled(7, "Parallel workers (0=auto)", ttk.Entry(opts, textvariable=self.var_workers, width=10))

        ttk.Checkbutton(opts, text="Normalize overlay", variable=self.var_norm).grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 2))

        # Output settings
        out = ttk.LabelFrame(left, text="Output", padding=8)
        out.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        out.columnconfigure(1, weight=1)

        self.var_out_dir = tk.StringVar(value=str(Path.cwd() / "ruby_rf_out"))
        self.var_save_plots = tk.BooleanVar(value=True)
        self.var_export_csv = tk.BooleanVar(value=True)

        # Plot style / labeling
        self.var_plot_title = tk.StringVar(value="Ruby fluorescence – {name}")
        self.var_xlabel = tk.StringVar(value="Wavelength (nm)")
        self.var_ylabel = tk.StringVar(value="Intensity")
        self.var_data_label_tpl = tk.StringVar(value="{name} data")
        self.var_fit_label_tpl = tk.StringVar(value="{name} fit")
        self.var_show_data = tk.BooleanVar(value=True)
        self.var_show_fit = tk.BooleanVar(value=False)
        self.var_fit_same_color = tk.BooleanVar(value=True)
        self.var_fit_dashed = tk.BooleanVar(value=True)

        ttk.Label(out, text="Output folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(out, textvariable=self.var_out_dir).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(out, text="Browse...", command=self._browse_outdir).grid(row=0, column=2, sticky="ew")

        ttk.Checkbutton(out, text="Save per-file plot (PNG)", variable=self.var_save_plots).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(out, text="Export summary CSV", variable=self.var_export_csv).grid(row=2, column=0, columnspan=3, sticky="w")

        # Plot styling / labels
        style = ttk.LabelFrame(left, text="Plot Style", padding=8)
        style.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        style.columnconfigure(1, weight=1)

        ttk.Label(style, text="Plot title").grid(row=0, column=0, sticky="w")
        ttk.Entry(style, textvariable=self.var_plot_title).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(6, 0))

        ttk.Label(style, text="X label").grid(row=1, column=0, sticky="w")
        ttk.Entry(style, textvariable=self.var_xlabel).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(6, 0))

        ttk.Label(style, text="Y label").grid(row=2, column=0, sticky="w")
        ttk.Entry(style, textvariable=self.var_ylabel).grid(row=2, column=1, columnspan=2, sticky="ew", padx=(6, 0))

        ttk.Label(style, text="Data label template").grid(row=3, column=0, sticky="w")
        ttk.Entry(style, textvariable=self.var_data_label_tpl).grid(row=3, column=1, columnspan=2, sticky="ew", padx=(6, 0))

        ttk.Label(style, text="Fit label template").grid(row=4, column=0, sticky="w")
        ttk.Entry(style, textvariable=self.var_fit_label_tpl).grid(row=4, column=1, columnspan=2, sticky="ew", padx=(6, 0))

        ttk.Checkbutton(style, text="Show data curves", variable=self.var_show_data).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(style, text="Show fitted curves (per dataset)", variable=self.var_show_fit).grid(row=6, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(style, text="Fit uses same color as data", variable=self.var_fit_same_color).grid(row=7, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(style, text="Fit line dashed", variable=self.var_fit_dashed).grid(row=8, column=0, columnspan=3, sticky="w")

        style_btns = ttk.Frame(style)
        style_btns.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        style_btns.columnconfigure((0, 1), weight=1)
        ttk.Button(style_btns, text="Set color for selected...", command=self._set_color_for_selected).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(style_btns, text="Clear custom colors", command=self._clear_colors).grid(row=0, column=1, sticky="ew")

        # Actions
        act = ttk.Frame(left)
        act.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        act.columnconfigure((0, 1), weight=1)

        ttk.Button(act, text="Analyze All", command=self._analyze_all).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(act, text="Plot Selected Overlay", command=self._plot_selected_overlay).grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        # Right: plot + results table + log
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=0)
        right.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(self.var_xlabel.get())
        self.ax.set_ylabel(self.var_ylabel.get())

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky="ne")

        # Results table
        table_frame = ttk.LabelFrame(right, text="Results", padding=6)
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = (
            "file",
            "ok",
            "primary_P",
            "primary_dP",
            "peak",
            "center",
            "dcenter",
            "P",
            "dP",
            "redchi",
        )
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("file", width=320, anchor="w")
        self.tree.column("ok", width=40, anchor="center")
        self.tree.column("primary_P", width=80, anchor="e")
        self.tree.column("primary_dP", width=80, anchor="e")
        self.tree.column("peak", width=45, anchor="e")
        self.tree.column("center", width=80, anchor="e")
        self.tree.column("dcenter", width=80, anchor="e")
        self.tree.column("P", width=80, anchor="e")
        self.tree.column("dP", width=80, anchor="e")
        self.tree.column("redchi", width=80, anchor="e")

        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")

        # Log + progress
        bottom = ttk.Frame(right)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.status = ttk.Label(bottom, text="Ready")
        self.status.grid(row=0, column=1, sticky="e")

    # ---- File management ----

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select spectra files",
            filetypes=[("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")]
        )
        if not paths:
            return
        self._extend_files(paths)

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(title="Select folder containing spectra")
        if not d:
            return
        folder = Path(d)
        # common: txt/dat/csv
        found = []
        for ext in ("*.txt", "*.dat", "*.csv"):
            found.extend([str(p) for p in folder.glob(ext)])
        found = sorted(set(found))
        if not found:
            messagebox.showinfo("No files", "No *.txt/*.dat/*.csv files found in the selected folder.")
            return
        self._extend_files(found)

    def _extend_files(self, paths: Sequence[str]) -> None:
        added = 0
        for p in paths:
            p = str(Path(p))
            if p not in self._files:
                self._files.append(p)
                self.file_list.insert(tk.END, p)
                added += 1
        self.status.configure(text=f"Added {added} file(s)")

    def _remove_selected(self) -> None:
        sel = list(self.file_list.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            p = self.file_list.get(idx)
            self.file_list.delete(idx)
            if p in self._files:
                self._files.remove(p)
            self._spectra_cache.pop(p, None)
            self._results.pop(p, None)
            self._color_map.pop(p, None)
        self.status.configure(text="Removed selected")

    def _clear_files(self) -> None:
        self.file_list.delete(0, tk.END)
        self._files.clear()
        self._spectra_cache.clear()
        self._results.clear()
        self._color_map.clear()
        self._clear_tree()
        self._clear_plot()
        self.status.configure(text="Cleared")

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.var_out_dir.set(d)

    def _set_color_for_selected(self) -> None:
        sel = list(self.file_list.curselection())
        if not sel:
            messagebox.showinfo("Select files", "Select one or more files in the list first.")
            return
        _rgb, hexcol = colorchooser.askcolor(title="Choose line color")
        if not hexcol:
            return
        for i in sel:
            fp = self.file_list.get(i)
            self._color_map[fp] = hexcol
        self.status.configure(text=f"Set color for {len(sel)} file(s)")

    def _clear_colors(self) -> None:
        self._color_map.clear()
        self.status.configure(text="Cleared custom colors")

    # ---- Plotting ----

    def _clear_plot(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel(self.var_xlabel.get())
        self.ax.set_ylabel(self.var_ylabel.get())
        self.canvas.draw_idle()

    def _plot_selected_overlay(self) -> None:
        sel = list(self.file_list.curselection())
        if not sel:
            messagebox.showinfo("Select files", "Select one or more files in the list first.")
            return

        files = [self.file_list.get(i) for i in sel]
        cfg = self._cfg_from_ui()

        title_tpl = self.var_plot_title.get().strip() or "Overlay"
        xlabel = self.var_xlabel.get().strip() or "Wavelength (nm)"
        ylabel = self.var_ylabel.get().strip() or "Intensity"
        data_tpl = self.var_data_label_tpl.get().strip() or "{name} data"
        fit_tpl = self.var_fit_label_tpl.get().strip() or "{name} fit"
        show_data = bool(self.var_show_data.get())
        show_fit = bool(self.var_show_fit.get())
        fit_same_color = bool(self.var_fit_same_color.get())
        fit_dashed = bool(self.var_fit_dashed.get())

        self.ax.clear()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title_tpl.replace("{name}", "Overlay"))

        roi = (cfg.roi_min_nm, cfg.roi_max_nm)

        for fp in files:
            name = Path(fp).stem
            x, y_raw = self._get_spectrum(fp)
            norm_factor = float(np.max(y_raw)) if (cfg.normalize_overlay and np.max(y_raw) > 0) else 1.0
            y = y_raw / norm_factor

            color = self._color_map.get(fp)
            data_label = data_tpl.format(name=name)
            fit_label = fit_tpl.format(name=name)

            if show_data:
                self.ax.plot(x, y, linewidth=1.0, label=data_label, color=color)

            if show_fit:
                try:
                    x_dense, y_dense = self._fit_curve_for_overlay(x, y_raw, roi, cfg)
                    y_dense = y_dense / norm_factor
                    ls = "--" if fit_dashed else "-"
                    fit_color = color if (fit_same_color and color) else None
                    self.ax.plot(x_dense, y_dense, linewidth=1.5, linestyle=ls, label=fit_label, color=fit_color)
                except Exception:
                    continue

        self.ax.set_xlim(cfg.roi_min_nm, cfg.roi_max_nm)
        self.ax.legend(fontsize=8, loc="best")
        self.canvas.draw_idle()
        self.status.configure(text=f"Overlay: {len(files)} file(s)")

    def _get_spectrum(self, fp: str) -> Tuple[np.ndarray, np.ndarray]:
        if fp in self._spectra_cache:
            return self._spectra_cache[fp]
        x, y = _parse_two_column_numeric_lines(Path(fp))
        self._spectra_cache[fp] = (x, y)
        return x, y

    def _fit_curve_for_overlay(
        self,
        x: np.ndarray,
        y: np.ndarray,
        roi: Tuple[float, float],
        cfg: AnalysisConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a fitted curve for plotting (no caching; fast enough for a few files)."""
        mask = (x >= roi[0]) & (x <= roi[1])
        x_roi = x[mask]
        y_roi = y[mask]
        guesses = detect_top_n_peaks(x_roi, y_roi, cfg.n_peaks, cfg.peak_distance, cfg.peak_prominence)
        if not guesses:
            raise ValueError("No peaks detected for fit")
        x_dense, y_dense, _, _ = fit_multi_voigt_lmfit(x, y, roi, guesses, cfg)
        return x_dense, y_dense

    # ---- Batch analysis ----

    def _cfg_from_ui(self) -> AnalysisConfig:
        try:
            roi_min = float(self.var_roi_min.get())
            roi_max = float(self.var_roi_max.get())
            n_peaks = int(float(self.var_n_peaks.get()))
            prom = float(self.var_prom.get())
            dist = int(float(self.var_dist.get()))
            lam_ref = float(self.var_lambda_ref.get())
            lam_ref_err = float(self.var_lambda_ref_err.get())
            workers = int(float(self.var_workers.get()))
        except Exception:
            raise ValueError("Invalid numeric setting(s).")

        cfg = AnalysisConfig(
            roi_min_nm=min(roi_min, roi_max),
            roi_max_nm=max(roi_min, roi_max),
            n_peaks=max(1, min(n_peaks, 5)),
            peak_prominence=max(0.0, prom),
            peak_distance=max(1, dist),
            lambda_ref_nm=lam_ref,
            lambda_ref_err_nm=max(0.0, lam_ref_err),
            normalize_overlay=bool(self.var_norm.get()),
        )
        # stash workers separately (GUI-only)
        self._workers = max(0, workers)
        return cfg

    def _analyze_all(self) -> None:
        if not self._files:
            messagebox.showinfo("No files", "Add files first.")
            return
        try:
            cfg = self._cfg_from_ui()
        except Exception as e:
            messagebox.showerror("Settings error", str(e))
            return

        out_dir = Path(self.var_out_dir.get()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = make_run_stamp()

        self._clear_tree()
        self.progress.configure(maximum=len(self._files), value=0)
        self.status.configure(text=f"Analyzing... run {run_stamp}")

        # Run in background to keep GUI responsive
        import threading

        def worker() -> None:
            try:
                results = run_batch(self._files, cfg, workers=getattr(self, "_workers", 0))
                # store
                self._results = {r.filepath: r for r in results}

                # export CSV with timestamp so old runs are not overwritten
                results_csv_path = None
                if self.var_export_csv.get():
                    results_csv_path = timestamped_output_path(out_dir, "ruby_rf_results.csv", run_stamp)
                    export_results_csv(results, results_csv_path)

                # plots with the same timestamp as the CSV
                if self.var_save_plots.get():
                    self._save_plots(results, cfg, out_dir, run_stamp=run_stamp)

                # update UI
                status_msg = f"Done: {sum(r.ok for r in results)}/{len(results)} OK"
                if results_csv_path is not None:
                    status_msg += f" | CSV: {results_csv_path.name}"
                self.after(0, lambda: self._populate_tree(results))
                self.after(0, lambda msg=status_msg: self.status.configure(text=msg))

            except Exception as ex:
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("Batch error", f"{ex}\n\n{tb}"))
                self.after(0, lambda: self.status.configure(text="Error"))
            finally:
                self.after(0, lambda: self.progress.configure(value=0))

        threading.Thread(target=worker, daemon=True).start()

    def _save_plots(self, results: Sequence[FileResult], cfg: AnalysisConfig, out_dir: Path, run_stamp: Optional[str] = None) -> None:
        # Create plots without blocking UI too much; do it in the worker thread.
        title_tpl = self.var_plot_title.get().strip() or "Ruby fluorescence – {name}"
        xlabel = self.var_xlabel.get().strip() or "Wavelength (nm)"
        ylabel = self.var_ylabel.get().strip() or "Intensity"
        data_tpl = self.var_data_label_tpl.get().strip() or "{name} data"
        fit_tpl = self.var_fit_label_tpl.get().strip() or "{name} fit"
        show_fit = bool(self.var_show_fit.get())
        fit_same_color = bool(self.var_fit_same_color.get())
        fit_dashed = bool(self.var_fit_dashed.get())

        roi = (cfg.roi_min_nm, cfg.roi_max_nm)
        if run_stamp is None:
            run_stamp = make_run_stamp()

        for i, r in enumerate(results, start=1):
            self.after(0, lambda v=i: self.progress.configure(value=v))
            if not r.ok:
                continue
            try:
                fp = r.filepath
                name = Path(fp).stem
                x, y_raw = self._get_spectrum(fp)
                color = self._color_map.get(fp)

                fig = Figure(figsize=(7, 5), dpi=120)
                ax = fig.add_subplot(111)

                # normalize if requested (use same normalization as overlay)
                norm_factor = float(np.max(y_raw)) if (cfg.normalize_overlay and np.max(y_raw) > 0) else 1.0
                y = y_raw / norm_factor

                ax.plot(x, y, linewidth=1.0, label=data_tpl.format(name=name), color=color)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title_tpl.format(name=name))
                ax.set_xlim(cfg.roi_min_nm, cfg.roi_max_nm)

                if show_fit:
                    try:
                        x_dense, y_dense = self._fit_curve_for_overlay(x, y_raw, roi, cfg)
                        y_dense = y_dense / norm_factor
                        ls = "--" if fit_dashed else "-"
                        fit_color = color if (fit_same_color and color) else None
                        ax.plot(x_dense, y_dense, linewidth=1.5, linestyle=ls, label=fit_tpl.format(name=name), color=fit_color)
                    except Exception:
                        pass

                # annotate peaks from existing analysis result
                txt_lines = []
                for p in r.peaks:
                    txt_lines.append(
                        f"Peak {p.peak_index}: {p.center_nm:.5f}±{p.center_err_nm:.5f} nm | "
                        f"{p.pressure_gpa:.3f}±{p.pressure_err_gpa:.3f} GPa"
                    )
                if txt_lines:
                    ax.text(
                        0.02,
                        0.98,
                        "\n".join(txt_lines[:6]),
                        transform=ax.transAxes,
                        va="top",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                    )
                ax.legend(fontsize=9, loc="best")

                out_png = timestamped_output_path(out_dir, name + ".png", run_stamp)
                fig.savefig(out_png, dpi=300)
            except Exception:
                continue

    def _populate_tree(self, results: Sequence[FileResult]) -> None:
        self._clear_tree()
        for r in results:
            if not r.peaks:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        Path(r.filepath).name,
                        "Y" if r.ok else "N",
                        f"{r.primary_pressure_gpa:.3f}" if r.primary_pressure_gpa is not None else "",
                        f"{r.primary_pressure_err_gpa:.3f}" if r.primary_pressure_err_gpa is not None else "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        f"{r.redchi:.3g}" if r.redchi is not None else "",
                    ),
                )
            else:
                for p in r.peaks:
                    self.tree.insert(
                        "",
                        tk.END,
                        values=(
                            Path(r.filepath).name,
                            "Y" if r.ok else "N",
                            f"{r.primary_pressure_gpa:.3f}" if r.primary_pressure_gpa is not None else "",
                            f"{r.primary_pressure_err_gpa:.3f}" if r.primary_pressure_err_gpa is not None else "",
                            p.peak_index,
                            f"{p.center_nm:.5f}",
                            f"{p.center_err_nm:.5f}",
                            f"{p.pressure_gpa:.3f}",
                            f"{p.pressure_err_gpa:.3f}",
                            f"{r.redchi:.3g}" if r.redchi is not None else "",
                        ),
                    )

    def _clear_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)


# -----------------------------
# Batch runner (GUI + CLI)
# -----------------------------

def run_batch(paths: Sequence[str], cfg: AnalysisConfig, workers: int = 0) -> List[FileResult]:
    paths = [str(Path(p)) for p in paths]
    if not paths:
        return []

    # workers: 0 => auto-ish; 1 => sequential
    if workers <= 0:
        # Keep it conservative for Windows laptops
        workers = min(4, (os.cpu_count() or 2))

    if workers <= 1 or len(paths) < 3:
        return [analyze_one_file(p, cfg) for p in paths]

    # ProcessPool for real speedup (fits are CPU-heavy)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: List[FileResult] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyze_one_file, p, cfg): p for p in paths}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(FileResult(filepath=futs[fut], ok=False, message=str(e), peaks=[]))

    # stable ordering like input
    by_path = {r.filepath: r for r in results}
    return [by_path.get(p, FileResult(filepath=p, ok=False, message="Missing result", peaks=[])) for p in paths]


# -----------------------------
# CLI
# -----------------------------

def cli_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Ruby RF batch analyzer (no GUI)")
    ap.add_argument("--cli", action="store_true", help="Run in CLI mode")
    ap.add_argument("--input", type=str, required=True, help="Input folder (or single file)")
    ap.add_argument("--glob", type=str, default="*.txt", help="Glob pattern when input is a folder")
    ap.add_argument("--out", type=str, required=True, help="Output folder")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    ap.add_argument("--roi-min", type=float, default=685.0)
    ap.add_argument("--roi-max", type=float, default=710.0)
    ap.add_argument("--n-peaks", type=int, default=2)
    ap.add_argument("--prominence", type=float, default=0.3)
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--lambda-ref", type=float, default=693.88947)
    ap.add_argument("--lambda-ref-err", type=float, default=0.05978)
    ns = ap.parse_args(list(argv))

    cfg = AnalysisConfig(
        roi_min_nm=min(ns.roi_min, ns.roi_max),
        roi_max_nm=max(ns.roi_min, ns.roi_max),
        n_peaks=max(1, min(ns.n_peaks, 5)),
        peak_prominence=max(0.0, ns.prominence),
        peak_distance=max(1, ns.distance),
        lambda_ref_nm=ns.lambda_ref,
        lambda_ref_err_nm=max(0.0, ns.lambda_ref_err),
    )

    inp = Path(ns.input).expanduser()
    if inp.is_dir():
        files = sorted(str(p) for p in inp.glob(ns.glob))
    else:
        files = [str(inp)]
    out_dir = Path(ns.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = make_run_stamp()
    results = run_batch(files, cfg, workers=ns.workers)

    results_csv_path = timestamped_output_path(out_dir, "ruby_rf_results.csv", run_stamp)
    export_results_csv(results, results_csv_path)

    ok = sum(r.ok for r in results)
    print(f"Done: {ok}/{len(results)} OK. Wrote: {results_csv_path}")
    # Also dump a small summary with the same timestamp
    summary_path = timestamped_output_path(out_dir, "ruby_rf_primary_summary.csv", run_stamp)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "primary_pressure_gpa", "primary_pressure_err_gpa", "ok", "message"])
        for r in results:
            w.writerow([r.filepath, r.primary_pressure_gpa, r.primary_pressure_err_gpa, r.ok, r.message])
    print(f"Wrote: {summary_path}")
    return 0


def main() -> int:
    # If user passes --cli, go CLI; otherwise launch GUI.
    if "--cli" in sys.argv:
        return cli_main(sys.argv[1:])

    app = RubyRFApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
