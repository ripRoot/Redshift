# Redshift Estimator via Cross-Correlation

This project estimates redshift by performing normalized cross-correlation between observed spectra and synthetic emission-line templates. It supports Sloan Digital Sky Survey (SDSS) spectra and allows you to select from templates for comparison. It is designed for astronomers and astrophysics researchers who need to estimate redshifts of celestial objects based on SDSS spectral data. It allows for fast, automated redshift estimation using prebuilt emission-line templates and normalized cross-correlation techniques. The output is the best-fit template, redshift value, and a similarity score.


To be added + Known Issues:
>Some SDSS templates fail to correlate correctly.

>REALLY poor time complexity. Will look into improving. (again) 

>No intergrated plotting yet.

>Paper+Slides

### Requirements
Before using the tool, clone the repo and install dependencies:
```bash
git clone https://github.com/ripRoot/Redshift.git
```
```bash
cd Redshift
```
```bash
pip install -r requirements.txt
```

### Project Structure Overview
````markdown
Redshift/
├── src/
│ ├── estimate_redshift.py # CLI tool for redshift estimation
│ ├── redshift_cross_correlation.py # Core algorithm
│ └── redshift_cc_sdss.py # SDSS spectrum loader + templates
├── spectra/ # Downloaded SDSS spectra
├── template_spectra/ # Prebuilt emission-line templates from SDSS
├── notebooks/ # Jupyter notebooks for interactive testing
│ └── redshift_cross_correlation_simulation.ipynb # Example notebook 
├── README.md
├── requirements.txt
└── LICENSE
````

### Getting Started

After installing the requirements, run this to download a spectrum from SDSS, match it to a template, and estimate redshift:
```bash
cd Redshift/src
```
```bash
python estimate_redshift.py --plate 266 --mjd 51602 --fiber 9
```
Optional parameters: 
>--type to switch between GALAXY, QSO, STAR. Default is ALL.

Example output:
```bash
Redshift Estimation Complete:
   Template index: 004
   Estimated redshift (z): 0.13600
   Correlation score: 5205.380
```
## Notebooks

All code used in the notebooks calls functions from `.py` files in `src/`. This ensures reproducibility and avoids embedding logic in cells:

```python
import sys
```
```python
sys.path.append('../src') # Adjust path if needed
```
```python
import redshift_cross_correlation as mrc
```
```python
import redshift_cc_sdss as crs
```


###  Load Template Spectra
You can load prebuilt emission lines in two ways:
```python
template_spectra = crs.get_all_template_spectra()
```
OR
```python
template_spectra = crs.get_template_spectra(type_str='GALAXY') # Options: GALAXY, QSO, STAR
```


### Load an Observed Spectrum
From SDSS:
```python
flux, wavelength = crs.get_spectrum(plate=266, mjd=51602, fiberID=9)
```
Or, if you want a default spectrum:
```python
flux, wavelength = crs.get_spectrum()
```


### Estimate Redshift
Perform redshift estimation via cross-correlation:
```python
template, redshift, correlation = mrc.cross_correlate_redshift(wavelength, flux, template_spectra)
```
This returns the best-fit template, best-fit redshift, and the correlation score. 

## Resources

Search for SDSS Spectra: https://dr17.sdss.org/optical/spectrum/view
Use this tool to look up spectra by plate, MJD, and fiber ID from SDSS DR17.


SDSS Template Source: https://classic.sdss.org/dr6/algorithms/spectemplates/index.php
The emission-line template spectra used in this project were obtained from this SDSS archive.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

