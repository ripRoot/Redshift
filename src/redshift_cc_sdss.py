from astropy.io import fits
import numpy as np
from astroquery.sdss import SDSS
import os
import redshift_cross_correlation as rcc

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

def get_template_spectra(type_str):
    template_dir = "../template_spectra/"
    
    # Define template number ranges by type
    type_ranges = {
        "STAR": range(0, 24),
        "GALAXY": range(24, 30),
        "QSO": range(30, 33),
    }
    
    if type_str not in type_ranges:
        raise ValueError(f"Unknown type '{type_str}'. Valid options: {list(type_ranges.keys())}")
    
    selected_indices = type_ranges[type_str]
    
    # Exclude 5 anyway if you want (optional, remove if not needed)
    selected_indices = [i for i in selected_indices]
    
    template_spectra = []
    
    for i in selected_indices:
        filename = f"spDR2-{i:03d}.fit"
        filepath = os.path.join(template_dir, filename)
        
        with fits.open(filepath) as hdul:
            data = hdul[0].data   
            header = hdul[0].header
            
            # Adjust flux scale if needed
            flux = data[0] 
            coeff0 = header['COEFF0']
            coeff1 = header['COEFF1']
        
            npix = flux.size
            pixel_indices = np.arange(npix)
            loglam = coeff0 + coeff1 * pixel_indices
            wavelength = 10**loglam
            
            template_spectra.append([wavelength, flux])
    
    return template_spectra

def get_all_template_spectra(template_dir="../template_spectra/"):
    """
    Load all template spectra from the specified directory.
    Assumes filenames are of the form 'spDR2-XXX.fit'.
    """
    template_spectra = []

    for filename in sorted(os.listdir(template_dir)):
        if filename.startswith("spDR2-") and filename.endswith(".fit"):
            filepath = os.path.join(template_dir, filename)

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
                print(f"Warning: Skipping {filename} due to error: {e}")

    return template_spectra

