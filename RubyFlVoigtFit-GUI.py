# helper_voigt_rf.py
# Python 3.7 / Win7 compatible
from typing import Tuple, Optional, List, Dict
import os
import re
import warnings

#update 2 elements for GUI
import os
import tkinter as tk
from tkinter import filedialog, messagebox



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.signal import find_peaks
from lmfit.models import VoigtModel, ConstantModel
from lmfit.parameter import Parameters

# suppress a specific lmfit warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Using UFloat objects with std_dev==0 may give unexpected results."
)


def read_two_column_txt(filepath: str) -> pd.DataFrame:
    """
    Reads a two-column whitespace-delimited text file,
    skipping the first 3 header lines.
    """
    return pd.read_csv(
        filepath,
        skiprows=3,
        sep=r'\s+',
        header=None,
        names=["Wavelength (nm)", "Counts"]
    )


def RF_peak_to_GPa_conversion(
        lambda_pk: float,
        lambda_ref: float = 693.88947
    ) -> float:
    #Convert ruby‐fluorescence peak shift to GPa.
    delta = lambda_pk - lambda_ref
    p = (1870 * (delta / lambda_ref)) * (1 + 5.63 * (delta / lambda_ref))
    return p


def RF_GPA_error(
        lambda_pk: float,
        lambda_pk_error: Optional[float],
        lambda_ref: float = 693.07943,
        lambda_ref_err: float = 0.05978
    ) -> float:
    #Estimate error on pressure from uncertainties in peak positions.
    # conservative estimate if no error provided
    if not lambda_pk_error:
        lambda_pk_error = 0.01
    if lambda_ref_err == 0:
        lambda_ref_err = 0.01

    delta = lambda_pk - lambda_ref
    delta_err = lambda_pk_error + lambda_ref_err

    # propagation of error: Δλ/λ_ref
    rel_err = (delta / lambda_ref) * np.sqrt(
        (delta_err / delta)**2 + (lambda_ref_err / lambda_ref)**2
    )
    return 1870 * rel_err


def detect_top_n_peaks(
        x: np.ndarray,
        y: np.ndarray,
        n_peaks: int = 3,
        distance: int = 2,
        prominence: float = 5.0
    ) -> List[Dict]:
    #Find the top-n most prominent peaks in (x, y).Returns initial guesses for Voigt fit.
    peaks, props = find_peaks(
        y,
        distance=distance,
        prominence=prominence
    )
    prominences = props["prominences"]
    # sort descending and take top n
    idx = np.argsort(prominences)[-n_peaks:][::-1]
    top = peaks[idx]

    guesses = []
    for p in top:
        guesses.append({
            "center": float(x[p]),
            "amplitude": float(y[p]),
            "sigma": 0.6,
            "gamma": 0.6
        })
    return guesses


def fit_multi_voigt_lmfit(
        data: pd.DataFrame,
        xlim: Tuple[float, float],
        peak_guesses: List[Dict]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Parameters]]:
    #Fit multiple Voigt profiles plus a constant background over the wavelength range xlim.
    x = data["Wavelength (nm)"].values
    y = data["Counts"].values

    # mask to region of interest
    mask = (x >= xlim[0]) & (x <= xlim[1])
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size == 0:
        return None

    # build model: constant + sum of Voigt
    model = ConstantModel(prefix='c_')
    params = model.make_params(c_c=40.0)

    for i, guess in enumerate(peak_guesses):
        prefix = f"v{i}_"
        vmod = VoigtModel(prefix=prefix)
        model += vmod
        params.update(vmod.make_params())
        params[f"{prefix}center"].set(value=guess["center"])
        params[f"{prefix}amplitude"].set(value=guess["amplitude"], min=0)
        params[f"{prefix}sigma"].set(value=guess["sigma"], min=0.1, max=2)
        params[f"{prefix}gamma"].set(value=guess["gamma"], min=0.1, max=2)

    result = model.fit(y_fit, params, x=x_fit)

    # generate a smooth curve
    x_dense = np.linspace(xlim[0], xlim[1], 500)
    y_dense = model.eval(result.params, x=x_dense)

    return x_dense, y_dense, result.params


def plot_wavelength_counts(
        data: pd.DataFrame,
        xlim: Optional[Tuple[float, float]] = None,
        fit_voigt: bool = False,
        n_peaks: int = 3,
        filename: str = ""
    ) -> Tuple[Figure, Axes]:

    #Scatter-plot the data, optionally fit Voigt peaks in [xlim].

    fig, ax = plt.subplots(figsize=(8, 6))
    x = data["Wavelength (nm)"]
    y = data["Counts"]

    ax.scatter(x, y, marker='o', label='Data')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Counts")

    title = os.path.splitext(os.path.basename(filename))[0]
    ax.set_title(f"Ruby Fluorescence Peak Fit: {title}")

    if xlim:
        ax.set_xlim(xlim)

    if fit_voigt and xlim:
        # detect & fit
        mask = (x >= xlim[0]) & (x <= xlim[1])
        guesses = detect_top_n_peaks(
            x[mask].values,
            y[mask].values,
            n_peaks=n_peaks,
            prominence=0.3
        )
        fit = fit_multi_voigt_lmfit(data, xlim, guesses)
        if fit:
            x_f, y_f, params = fit
            ax.plot(x_f, y_f, 'r-', label='Voigt Fit')
            ax.legend()

            # annotate peaks & pressures
            notes = []
            for i in range(len(guesses)):
                pfx = f"v{i}_"
                center = params[pfx + "center"].value
                err_raw = params[pfx + "center"].stderr
                err = err_raw if err_raw and err_raw > 0 else 0.01

                P = RF_peak_to_GPa_conversion(center)
                P_err = RF_GPA_error(center, err)

                if i == 0:
                    notes.append(
                        rf"$\bf{{Peak\,{i+1}\ xc:}}$ "
                        f"{center:.5f}±{err:.5f}\n"
                        rf"$\bf{{Pressure:}}$ {P:.3f}±{P_err:.3f} GPa"
                    )
                else:
                    notes.append(
                        f"Peak {i+1}: {center:.5f}±{err:.5f}\n"
                        f"Pressure: {P:.3f}±{P_err:.3f} GPa"
                    )
            ax.text(
                0.60, 0.95,
                "\n\n".join(notes),
                transform=ax.transAxes,
                verticalalignment='top',
                bbox={'facecolor':'white','alpha':0.7}
            )

    return fig, ax

# … all your imports and helper functions (read_two_column_txt, plot_…, etc.) stay the same …

def get_filepath_from_user() -> str:
    path = input("Please enter the full path to your .txt data file: ").strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    return path

def get_filepath_via_gui() -> str:
    root = tk.Tk(); root.title("Select Ruby-RF Data File")
    base_var, sub_var, file_var = tk.StringVar(), tk.StringVar(), tk.StringVar()

    def browse_base():
        d = filedialog.askdirectory(title="Select base directory")
        if d: base_var.set(d)

    def on_ok():
        full = os.path.normpath(os.path.join(
            base_var.get().strip(),
            sub_var.get().strip(),
            file_var.get().strip()
        ))
        if not os.path.isfile(full):
            messagebox.showerror("File Not Found", full)
            return
        root.selected = full
        root.quit()

    # layout…
    tk.Label(root, text="1) Base directory:").grid(row=0, column=0)
    tk.Entry(root, textvariable=base_var, width=50).grid(row=0, column=1)
    tk.Button(root, text="Browse…", command=browse_base).grid(row=0, column=2)
    tk.Label(root, text="2) Sub-path (optional):").grid(row=1, column=0)
    tk.Entry(root, textvariable=sub_var, width=50).grid(row=1, column=1, columnspan=2)
    tk.Label(root, text="3) Filename:").grid(row=2, column=0)
    tk.Entry(root, textvariable=file_var, width=50).grid(row=2, column=1, columnspan=2)
    tk.Button(root, text="OK", command=on_ok).grid(row=3, column=1, pady=10)

    try:
        root.mainloop()
    finally:
        root.destroy()

    return getattr(root, "selected", None)

def choose_filepath() -> str:
    """
    Try the GUI first; on any failure or cancel, fall back to console.
    """
    try:
        fp = get_filepath_via_gui()
        if not fp:
            raise RuntimeError("No file selected in GUI")
        return fp
    except Exception as e:
        print(f"GUI picker failed ({e}), falling back to console input.")
        return get_filepath_from_user()

if __name__ == "__main__":
    filepath = choose_filepath()
    # … then your existing processing …
    df = read_two_column_txt(filepath)
    fig, ax = plot_wavelength_counts(
        df, xlim=(685.0, 710.0), fit_voigt=True, n_peaks=3, filename=filepath
    )
    plt.show()
    out_dir = os.path.dirname(filepath)
    base   = os.path.splitext(os.path.basename(filepath))[0]
    out_fp = os.path.join(out_dir, f"{base}.png")
    fig.savefig(out_fp, dpi=600)
    print("Saved figure to:", out_fp)
