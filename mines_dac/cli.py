import sys
import os
from mines_dac.analysis import plot_fitted_spectra
from mines_dac.file_dialog import select_file_exe, select_file_bkg
from mines_dac.validator import validate_spectra_file

import warnings
# Suppress specific FutureWarnings and UserWarnings
warnings.filterwarnings("ignore", message=".*delim_whitespace.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*std_dev==0.*", category=UserWarning)


reference_peak_value = 693.88947
reference_uncertainty = 0.07073

def print_startup_message():
    print("=" * 70)
    print("Welcome to the Mines DAC Pressure Analysis Tool!")
    print("=" * 70)
    print("\nPlease select an option:")
    print("\t1. Configure reference ruby fluorescence peak value and uncertainty (nm)")
    print("\t2. Select a spectra file to process")
    print("\t3. Exit\n")

def configure_defaults():
    global reference_peak_value, reference_uncertainty
    try:
        print("\t--- CONFIGURE REFERENCE VALUES ---")
        print(f"\tCurrent: Peak = {reference_peak_value} nm, Uncertainty = {reference_uncertainty} nm")
        reference_peak_value = float(input("\tEnter new reference peak value (nm): "))
        reference_uncertainty = float(input("\tEnter new default uncertainty (nm): "))
        print(f"\tUpdated: Peak = {reference_peak_value} nm, Uncertainty = {reference_uncertainty} nm\n")
    except ValueError:
        print("\n\tInvalid input. Using previous defaults.\n")

def process_file():
    while True:
        is_frozen = getattr(sys, 'frozen', False)
        if is_frozen:
            file_path = select_file_exe()
        else:
            file_path = select_file_bkg()
        if not file_path:
            print("No file selected.")
            break

        if not os.path.isfile(file_path):
            print("\tInvalid file path.")
            continue

        if validate_spectra_file(file_path):
            print(f"\n\tValid spectra file selected: {file_path}\n")
            fig, _ = plot_fitted_spectra(file_path, reference_peak_value, reference_uncertainty)
            plot_name = os.path.splitext(file_path)[0] + ".png"
            fig.savefig(plot_name, dpi=600)
            print(f"\tSpectra plotted successfully.")
            print(f"\tCurrent: Peak = {reference_peak_value} nm, Uncertainty = {reference_uncertainty} nm")
            print(f"\tPlot saved to: {plot_name}\n")
            break
        else:
            print("\n\tInvalid spectra file format.\n")
            retry = input("\n\tTry another file? (y/n): ").strip().lower()
            if retry != 'y':
                break

def main():
    while True:
        print_startup_message()
        choice = input("Enter your choice (1-3): ").strip()
        print()
        if choice == "1":
            configure_defaults()
        elif choice == "2":
            process_file()
        elif choice == "3":
            print("\tExiting the program...")
            sys.exit(0)
        else:
            print("\tInvalid choice. Please select 1, 2, or 3.\n")

if __name__ == "__main__":
    main()
