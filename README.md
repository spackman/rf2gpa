# Mines DAC Pressure Analysis Tool

A command-line interface (CLI) tool for analyzing ruby fluorescence spectra data and calculating pressure based on the DAC (diamond anvil cell) method.

## Features

- Configure reference ruby fluorescence peak value and uncertainty.
- Select a spectra file for processing.
- Fit the spectra using a Voigt model.
- Plot the fitted spectra and calculate pressure in GPa.
- Generates high-quality PNG plots of the spectra.

## Requirements

- Python 3.8 or higher
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `lmfit`

## Installation

To install the tool locally, clone the repository and install it via `pip`:

```bash
git clone https://github.com/yourusername/mines-dac-pressure.git
cd mines-dac-pressure
pip install -e .
``` 
Alternatively, download the executable for windows.  
  
## Usage  
  
Run the tool in your terminal:
```bash 
dac-pressure
```
 ### Options:
 1) **Configure Reference Values**  
 Set the referece ruby fluorescence peak value and uncertainty for this session. Default: 693.88947 ± 0.07073.  
 2) **Select Spectra File**  
 Choose a `.txt` or `.csv` file containing fluorescence spectra to process.  
 3) **Exit**  
 Exit the program.  
   
## License  
This project is licensed under the MIT license.  

## Icon Credit  
Icon credits to Cheng Ji (https://www.eurekalert.org/multimedia/599975) *Carnegie Institute For Science*  

## Authors  
* **Isaac Spackman**, *Colorado School of Mines*
* **Kacy Mendoza**, *Colorado School of Mines*  
---