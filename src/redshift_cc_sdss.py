from astropy.io import fits
import numpy as np
from astroquery.sdss import SDSS
import os

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

    #change that badboy to linear
    wavelength = 10**data['loglam']
    flux = data['flux']
    hdulist.close()

    return flux, wavelength

def save_spectrum_to_data(spec, filename):
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, '..', 'spectra')
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, filename)
    spec.writeto(output_path, overwrite=True)
    return output_path


def get_template_spectra(type_str, template_dir="../template_spectra/"):
    """
    Load selected template spectra by type, or all if 'ALL' is specified.
    Valid options for type_str: 'STAR', 'GALAXY', 'QSO', or 'ALL'
    """
    type_str = type_str.upper()
    
    type_ranges = {
        "STAR": range(0, 24),
        "GALAXY": range(24, 30),
        "QSO": range(30, 33),
        "ALL": range(0, 33),
    }
    
    if type_str not in type_ranges:
        raise ValueError(f"Unknown type '{type_str}'. Valid options: {list(type_ranges.keys())}")
    
    selected_indices = type_ranges[type_str]
    template_spectra = []

    for i in selected_indices:
        filename = f"spDR2-{i:03d}.fit"
        filepath = os.path.join(template_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Template file not found: {filename}")
            continue

        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data   
                header = hdul[0].header

                flux = data[0]
                coeff0 = header['COEFF0']
                coeff1 = header['COEFF1']

                npix = flux.size
                pixel_indices = np.arange(npix)
                loglam = coeff0 + coeff1 * pixel_indices
                wavelength = 10 ** loglam

                template_spectra.append([wavelength, flux])
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return template_spectra