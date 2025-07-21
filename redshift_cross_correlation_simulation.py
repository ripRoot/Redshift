import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.signal import medfilt


def redshift_main():    
    rest_lines = np.array([5000.0, 6000.0, 6500.0])   # H‑beta, [O III], H‑alpha-ish placeholders
    amps       = np.array([1.0,   0.9,   0.75])        # relative line strengths.
    
    
    wl = np.linspace(4800, 80000, 200000) # Changed it so that I can calculate up to 10 redshift, otherwise it slides off the side of the spectrum.

    # ------------------------------------
    # Apply a redshift to generate “data”
    # ------------------------------------

    z_true = np.random.uniform(0.0, 10)              # the unknown redshift we’ll try to recover
    observed_flux = build_shifted_template(wl, rest_lines, amps, z_true)


    # Add a touch of noise so it looks more realistic
    rng = np.random.default_rng(42)
    observed_flux += 0.05 * rng.normal(size=wl.size)

    # Normalized both observed_flux and the template.
    observed_flux_normalized = normalize_spectrum(observed_flux)

    # Greatest distance calculated using redshift is 10. 
    z_min, z_max = 0, 10 
    initial_pass = (z_min, z_max)


    best_z = calculate_redshift_iterative(
        observed_flux, rest_lines, wl, amps, initial_pass
        )

    print(f"The best redshift I found was {best_z}. The real redshift is {z_true}")

    print(f"My percent error is {abs((best_z - z_true)/z_true * 100)}.\n")

    return best_z, abs((best_z - z_true)/z_true * 100)
    

def gaussian(w, w0, amp, sigma=5.0):
    """
    Simple Gaussian line profile
    """
    return amp * np.exp(-(w - w0)**2 / (2.0 * sigma**2))

def build_shifted_template(wl, rest_lines, amps, z):

    '''ADDED'''
    shifted_lines = rest_lines * (1 + z)
    
    template_flux = np.full_like(wl, 0.01)

    for w0, amp in zip(shifted_lines, amps):
        template_flux += gaussian(wl, w0, amp)
    return normalize_spectrum(template_flux)


def normalize_spectrum(flux):
    '''
        GOAL: Remove the continuum (the smooth background light) to emphasize
        absorption & emission lines... (Not really required for this simulated data, but I read it
        was good to consider when calculating redshift of real spectra...)
    '''
    
    N = len(flux)

    kernel_fraction = 0.01 # 1% of the total length.. 
    
    # since our simulated len of flux is 7000 it will take the median of 70 points.
    kernel_size = int(N * kernel_fraction)

    # Make sure that the window size is odd, because without it no middle
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Traces the curve, or 'shape' of the Spectrum... focusing on smoothing the hills instead of the peaks.
    continum = medfilt(flux, kernel_size)

    # Subtracting the continuum leaves me with just the sharp lines. 
    flux_no_continum = flux - continum
    return (flux_no_continum - np.mean(flux_no_continum)) / np.std(flux_no_continum)

'''
def shift_and_interpolate(wl, flux_temp, z):
    
        This function shifts the wavelength by an assumed redshift value and then plots the flux values based on the shift.
        My interpolate function maps any wavelength back to a flux value based on the template.
    
    shifted_wl = wl * (1 + z)
    interp_func = interp1d(shifted_wl, flux_temp, bounds_error=False, fill_value=0)
    return interp_func(wl)
'''

def compute_correlate(observed, shifted_temp):
    '''
        Correlation. Described as below. the 'same' tag means that the output is the same as my input signal. Correlation is now centered. 
        returns the highest correlation point in the array.

        CHANGEED**
    '''
    observed_norm = normalize_spectrum(observed)
    observed_smooth = np.convolve(observed_norm, np.hanning(5)/np.sum(np.hanning(5)), mode='same')
    shifted_smooth = np.convolve(shifted_temp, np.hanning(5)/np.sum(np.hanning(5)), mode='same')
    return np.max(correlate(observed_norm, shifted_temp, mode='same'))


def calculate_redshift(assumed_redshifts, observed_flux, rest_lines, wl, amps):
    '''
        Takes every z value in the assumed redshifts, shifts the template with the rest_lines and gaussian to then cross correlate it with the observed spectrum. 
        It returns the scores (which is the cross correlation) and the redahift that has the highest cross correlation.
    '''
    scores = []
    for z in assumed_redshifts:
        template_flux = build_shifted_template(wl, rest_lines, amps, z)
        score = compute_correlate(observed_flux, template_flux)    
        scores.append(score)
    
    return scores, assumed_redshifts[np.argmax(scores)]


def calculate_redshift_iterative(observed_flux, rest_lines, wl, amps, initial_pass, passes=5):
    '''
        Iteratively narrows down the best redshift.
    '''    
    # unpack the tuple
    z_min, z_max = initial_pass
    best_z = np.inf

    
    for i in range(passes):
        num_steps = 400 if i == 0 else 150 # Reduced. Tried to make it more efficent. 
        
        # Log sampling for first pass, linear after
       
        # Avoid log 0 by a offset. Broad, logarithmic sweep...
        if i == 0:
            # Broad logarithmic sweep on first pass to better cover wide z range
            assumed_redshifts = np.exp(np.linspace(np.log(z_min + 1e-5), np.log(z_max + 1e-2), num_steps)) - 1e-5
        else:
            assumed_redshifts = np.linspace(z_min, z_max, num_steps)
        
        
        scores, best_z = calculate_redshift(
            assumed_redshifts, observed_flux, rest_lines, wl, amps
            )

        print(f"Pass {i+1}: best_z = {best_z} in range {z_min} to {z_max}")

        if i < 4:
            z_min, z_max = 0, 10
        else:
            narrow = max((z_max - z_min) / 10, 0.01)
            z_min = max(best_z - narrow, 0.01)
            z_max = best_z + narrow
    
    return best_z
