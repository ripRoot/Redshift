from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def plot_sdss_spectrum(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # shape (4, Npix)
        header = hdul[0].header
        
        # Extract flux: first row is the raw spectrum
        flux = data[0]

        # Extract log-wavelength coefficients from header
        coeff0 = header['COEFF0']  # e.g., log10(lambda_0)
        coeff1 = header['COEFF1']  # e.g., log10 step per pixel
        
        npix = flux.size
        pixel_indices = np.arange(npix)

        # Compute wavelength in Angstroms
        loglam = coeff0 + coeff1 * pixel_indices
        wavelength = 10 ** loglam

        # Plot
        plt.plot(wavelength, flux)
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Flux (10^-17 erg/cm^2/s/Angstrom)')
        plt.title(f'SDSS Spectrum from {fits_file}')
        plt.show()

# Usage
plot_sdss_spectrum('data/spDR2-023.fit')
