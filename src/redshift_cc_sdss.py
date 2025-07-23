from astropy.io import fits
#import matplotlib.pyplot as plt
import numpy as np
from astroquery.sdss import SDSS
import os
#import redshift_cross_correlation as redshift_cross_correlation


def get_spectrum(plate=2663, mjd=54234, fiberID=332):
    '''
        SDSS spectroscopic identifier 
    '''

    spectra = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberID)
    
    if spectra:
        print("Spectrum retrieved!")
        spectrum = get_flux_and_wl(spectra)
        return spectrum
    else:
        print("No spectrum found.")
        return None

def get_flux_and_wl(spectra):

    output_dir = save_spectrum_to_data(spectra[0], "sdss_spectrum.fits")
    hdulist = fits.open(output_dir)
    
    data = hdulist[1].data
    header0 = hdulist[0].header
    sdss_z = header0.get('Z', None)
    print(f"SDSS catalog redshift: {sdss_z}")

    #change that badboy to linear
    wavelength = 10**data['loglam']
    flux = data['flux']
    hdulist.close()

    return flux, wavelength

def save_spectrum_to_data(spec, filename):
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, filename)
    spec.writeto(output_path, overwrite=True)
    return output_path
