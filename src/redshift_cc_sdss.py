from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astroquery.sdss import SDSS
import redshift_cross_correlation as redshift_cross_correlation


def get_spectrum(plate=2663, mjd=54234, fiberID=332):
    '''
        SDSS spectroscopic identifier 
    '''

    spectra = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberID)
    
    if spectra:
        print("Spectrum retrieved!")
        return get_flux_and_wl(spectra)
    else:
        print("No spectrum found.")
        return None

def get_flux_and_wl(spectra):
    spectra[0].writeto("sdss_spectrum.fits", overwrite=True)
    hdulist = fits.open('sdss_spectrum.fits')
    data = hdulist[1].data
    header0 = hdulist[0].header
    sdss_z = header0.get('Z', None)
    print(f"SDSS catalog redshift: {sdss_z}")

    wavelength = 10**data['loglam']
    flux = data['flux']
    hdulist.close()
    return flux, wavelength

'''
def main():
    
        Main function to retrieve and plot SDSS spectrum
    
    # Get the spectrum data
    spectra = get_spectrum(plate=2663, mjd=54234, fiberID=332)
    flux, wavelength = get_flux_and_wl(spectra)

    print(flux, wavelength)
    # Assume galaxy
    rest_lines = [6563, 5007, 4861]  # Hα, [O III], Hβ

if __name__ == "__main__":
    main()
'''
