# Mines DAC Pressure Analysis Tool (rf2gpa)

Analyze ruby fluorescence spectra from a diamond anvil cell (DAC) and convert fitted peak positions into pressure (GPa).

This repository now ships a **single, self-contained core script** with a built-in Tkinter GUI:

- `rf2gpa.py` (GUI + optional CLI batch mode)

## Features

- Robust parsing of *most* two-column spectra files (handles noisy headers/footers better than fixed `skiprows`).
- Peak detection (`scipy.signal.find_peaks`) and **multi-Voigt** fitting (`lmfit`) inside a user-set Region of Interest.
- Overlay plotting (optionally normalized), optional per-dataset fitted curves.
- Batch analysis with progress + results table.
- Exports:
  - `ruby_rf_results.csv` (per-file + per-peak rows)
  - optional per-file `.png` plots

## Requirements

- Python 3.8+ (tested with Python 3.12)
- `numpy`, `scipy`, `pandas`, `matplotlib`, `lmfit`

Install deps:

```bash
pip install numpy scipy pandas matplotlib lmfit
```

## Run the GUI

From the project folder:

```bash
python rf2gpa.py
```

(Alternative convenience launcher)

```bash
python run_gui.py
```

## CLI batch mode (no GUI)

The same core file also supports a batch CLI mode:

```bash
python rf2gpa.py --cli \
  --input "./testfiles" \
  --glob "*.txt" \
  --out "./ruby_rf_out" \
  --workers 0
```

## Notes

- Default Region of Interest is 685–710 nm.
- The reference wavelength (`λ_ref`) and its uncertainty are editable in the GUI.
- The bundled `mines_dac/` package and `dac-pressure` CLI entrypoint are retained for backward compatibility,
  but **the new GUI script is the recommended workflow going forward**.

## License

MIT (see `LICENSE`).

## Icon Credit

Icon credits to Cheng Ji (Carnegie Institute for Science) (https://www.eurekalert.org/multimedia/599975)

## Authors

- Isaac Spackman (Colorado School of Mines)
- Kacy Mendoza (Colorado School of Mines)
- Vincent Castilow (Colorado School of Mines)
