from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astroquery.sdss import SDSS
import redshift_cross_correlation_simulation


def get_spectrum():
    '''
        SDSS spectroscopic identifier 
    '''
    plate = 2663
    mjd = 54234
    fiberID = 332

    spectra = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberID)

    if spectra:
        print("Spectrum retrieved!")
        spectra[0].writeto("sdss_spectrum.fits", overwrite=True)
    else:
        print("No spectrum found.")


    hdulist = fits.open('sdss_spectrum.fits')

    data = hdulist[1].data

    flux = data['flux']
    loglam = data['loglam']       # log10 of wavelength in Angstroms

    # Convert log-lambda to wavelength
    wavelength = 10**loglam
    
    hdulist.close()
    return flux, wavelength

def plot_spectrum(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(12,6))
    plt.plot(x, y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

xlabel = 'Wavelength (Angstrom)'
ylabel = 'Flux'
title = 'SDSS Spectrum'
y, x = get_spectrum()
plot_spectrum(x, y, xlabel, ylabel, title)

