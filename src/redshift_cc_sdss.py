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
        spectrum = get_flux_and_wl(spectra)
        return spectrum
    else:
        print("No spectrum found.")
        return None

def get_flux_and_wl(spectra):
    output_dir = save_spectrum_to_data(spectra[0], "sdss_spectrum.fits")
    hdulist = fits.open(output_dir)
    
    # Get flux and wavelength from HDU 1
    data = hdulist[1].data
    wavelength = 10 ** data['loglam']
    flux = data['flux']
    
    # Get redshift and class from HDU 2
    try:
        z_data = hdulist[2].data
        sdss_z = z_data['Z'][0]
        sdss_z_err = z_data['Z_ERR'][0]
        sdss_class = z_data['CLASS'][0]
        zwarning = z_data['ZWARNING'][0]
    except Exception as e:
        sdss_z = None

    hdulist.close()
    
    return flux, wavelength, sdss_z

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