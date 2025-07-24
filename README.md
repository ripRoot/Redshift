# Redshift Estimator via Cross-Correlation

Builds emission-line templates from given rest wavelengths and estimated line amplitudes. It does this through resampling spectra on a logarithmic wavelength grid for efficient multiplicative redshift handling, and uses normalized cross-correlation to estimate a best-fit redshift. 

Clone the repo and install dependencies:
<pre> git clone [githu](https://github.com/ripRoot/Redshift.git) 
cd Redshift
pip install -r requirements.txt </pre>

To be added in the future:
>Create a function to determine if absorption lines or emission lines are more prominent in the spectrum.

>Improve the find_amps function.
