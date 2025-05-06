from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting for CLI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from lmfit.models import VoigtModel, ConstantModel
from lmfit.parameter import Parameters
from scipy.signal import find_peaks
import os


def plot_fitted_spectra(
    file_path: str,
    reference_peak_value: float,
    reference_peak_uncertainty: float,
    xlim: Tuple[float, float] = (685, 710)
) -> Optional[Tuple[Figure, Axes]]:
    """
    Load and fit a spectra file, then plot the results with Voigt model peaks and annotations.

    Args:
        file_path (str): Path to the spectra file.
        reference_peak_value (float): Reference peak wavelength for pressure conversion.
        reference_peak_uncertainty (float): Uncertainty of the reference peak.
        xlim (Tuple[float, float]): X-axis range to fit and plot.

    Returns:
        Optional[Tuple[Figure, Axes]]: The matplotlib Figure and Axes, or None if fitting fails.
    """
    try:
        spectra = pd.read_csv(
            file_path, skiprows=3, delim_whitespace=True,
            header=None, names=["Wavelength (nm)", "Counts"]
        )
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    x = spectra["Wavelength (nm)"]
    y = spectra["Counts"]

    mask = (x >= xlim[0]) & (x <= xlim[1])
    x_fit_data = x[mask].values
    y_fit_data = y[mask].values

    peak_guesses = detect_top_n_peaks(x_fit_data, y_fit_data)

    fit_result = fit_multi_voigt_lmfit(x_fit_data, y_fit_data, peak_guesses)
    if not fit_result:
        print("Fitting failed. Aborting...")
        return None

    x_fit, y_fit, params = fit_result

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x_fit_data, y_fit_data, color='black', marker='o', label='Data')
    ax.plot(x_fit, y_fit, color='red', label='Voigt Fit')

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Counts")
    title_name = os.path.splitext(os.path.basename(file_path))[0]
    ax.set_title(f"Ruby Fluorescence Peak Fit: {title_name}")
    ax.legend(loc="upper left")

    annotations = []
    for i in range(3):
        prefix = f"v{i}_"
        center = params[prefix + "center"].value
        center_err = params[prefix + "center"].stderr or 0.0

        GPa = RF_peak_to_GPa_conversion(center, reference_peak_value)
        GPa_err = RF_GPa_error(center, center_err, reference_peak_value, reference_peak_uncertainty)

        if i == 0:
            annotations.append(
                rf"$\bf{{Peak\ {i+1}\ xc:}}$ {center:.5f} ± {center_err:.5f}" + "\n"
                rf"$\bf{{Pressure:}}$ {GPa:.3f} ± {GPa_err:.3f} GPa"
            )
        else:
            annotations.append(f"Peak {i+1} xc: {center:.5f} ± {center_err:.5f}")

    ax.text(0.98, 0.97, "\n\n".join(annotations), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            multialignment='left',
            bbox=dict(facecolor='white', alpha=0.7))
    
    fig.tight_layout()

    return fig, ax


def fit_multi_voigt_lmfit(
    x_fit_data: np.ndarray,
    y_fit_data: np.ndarray,
    peak_guesses: List[Dict[str, float]]
) -> Optional[Tuple[np.ndarray, np.ndarray, Parameters]]:
    """
    Fit multiple Voigt peaks plus a constant background to spectra data.

    Args:
        x_fit_data (np.ndarray): X data (wavelengths).
        y_fit_data (np.ndarray): Y data (counts).
        peak_guesses (List[Dict[str, float]]): List of initial parameter guesses for Voigt peaks.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, Parameters]]: Fitted x/y data and parameters, or None.
    """
    if len(x_fit_data) == 0 or len(peak_guesses) == 0:
        return None

    model = ConstantModel(prefix='c_')
    params = model.make_params(c_c=40.0)

    for i, guess in enumerate(peak_guesses):
        prefix = f"v{i}_"
        voigt = VoigtModel(prefix=prefix)
        model += voigt
        params.update(voigt.make_params())
        params[f"{prefix}center"].set(value=guess["center"])
        params[f"{prefix}amplitude"].set(value=guess["amplitude"], min=0)
        params[f"{prefix}sigma"].set(value=guess["sigma"], min=0.1, max=2)
        params[f"{prefix}gamma"].set(value=guess["gamma"], min=0.1, max=2)

    result = model.fit(y_fit_data, params, x=x_fit_data)
    x_dense = np.linspace(x_fit_data[0], x_fit_data[-1], 500)
    y_dense = model.eval(result.params, x=x_dense)

    return x_dense, y_dense, result.params


def detect_top_n_peaks(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int = 3,
    distance: int = 2,
    prominence: float = 0.3
) -> List[Dict[str, float]]:
    """
    Detect and return parameter guesses for the top N peaks in the data.

    Args:
        x (np.ndarray): X-axis data (wavelengths).
        y (np.ndarray): Y-axis data (counts).
        n_peaks (int): Number of peaks to detect.
        distance (int): Minimum distance between peaks.
        prominence (float): Minimum prominence of peaks.

    Returns:
        List[Dict[str, float]]: Peak parameter guesses for Voigt fitting.
    """
    peaks, properties = find_peaks(y, distance=distance, prominence=prominence)
    prominences = properties.get("prominences", [])
    if len(peaks) == 0 or len(prominences) == 0:
        return []

    top_indices = np.argsort(prominences)[-n_peaks:][::-1]
    top_peaks = peaks[top_indices]

    guesses = []
    for peak in top_peaks:
        center = x[peak]
        amplitude = y[peak]
        guesses.append({
            'center': center,
            'amplitude': amplitude,
            'sigma': 0.6,
            'gamma': 0.6
        })

    return guesses


def RF_peak_to_GPa_conversion(
    lambda_pk: float,
    lambda_ref: float
) -> float:
    """
    Convert the ruby fluorescence peak wavelength to pressure in GPa.

    Args:
        lambda_pk (float): Observed peak wavelength.
        lambda_ref (float): Reference peak wavelength.

    Returns:
        float: Pressure in GPa.
    """
    delta_lambda = lambda_pk - lambda_ref
    pressure = (1870 * (delta_lambda / lambda_ref)) * (1 + 5.63 * (delta_lambda / lambda_ref))
    return pressure


def RF_GPa_error(
    lambda_pk: float,
    lambda_pk_error: float,
    lambda_ref: float,
    lambda_ref_err: float
) -> float:
    """
    Estimate the uncertainty of the pressure in GPa due to peak position and reference error.

    Args:
        lambda_pk (float): Measured peak wavelength.
        lambda_pk_error (float): Error in measured peak.
        lambda_ref (float): Reference peak wavelength.
        lambda_ref_err (float): Error in reference peak.

    Returns:
        float: Estimated pressure uncertainty in GPa.
    """
    delta_lambda = lambda_pk - lambda_ref
    delta_lambda_err = lambda_pk_error + lambda_ref_err

    if delta_lambda == 0:
        return 0.0  # Avoid division by zero

    relative_error = abs(delta_lambda / lambda_ref * np.sqrt(
        (delta_lambda_err / delta_lambda) ** 2 +
        (lambda_ref_err / lambda_ref) ** 2
    ))

    pressure_error = 1870 * (relative_error + 5.63 * np.sqrt(2) * relative_error)
    return pressure_error
