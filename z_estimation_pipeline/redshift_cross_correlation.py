import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.signal import medfilt


def main():    
    rest_lines = np.array([5000.0, 6000.0, 6500.0])   # H‑beta, [O III], H‑alpha-ish placeholders
    amps       = np.array([1.0,   0.8,   0.6])        # relative line strengths.
    wl = np.linspace(4800, 9000, 7000)
    #Zeros_like returns an array of zeros 
    # with the same shape and size as a

    # Build rest‑frame template spectrum
    template_flux = np.zeros_like(wl)
    for w0, amp in zip(rest_lines, amps):
        template_flux += gaussian(wl, w0, amp)

    # ------------------------------------
    # Apply a redshift to generate “data”
    # ------------------------------------
    z_true = np.random.uniform(0.0, 0.37)              # the unknown redshift we’ll try to recover
    obs_lines = (1.0 + z_true) * rest_lines

    observed_flux = np.zeros_like(wl)
    for w0, amp in zip(obs_lines, amps):
        observed_flux += gaussian(wl, w0, amp)


    # Add a touch of noise so it looks more realistic
    rng = np.random.default_rng(42)
    observed_flux += 0.05 * rng.normal(size=wl.size)

    # a plot spectrum function.
    plot_spectrum(wl, observed_flux)

    # Mark the *rest* wavelengths (dotted) and observed wavelengths (dash‑dot)
    for w_rest, w_obs in zip(rest_lines, obs_lines):
        plt.axvline(w_rest, linestyle=':', alpha=0.4)        #plt.axvline(w_obs,  linestyle='-.', alpha=0.4)

        
    # Normalized both observed_flux and the template.
    observed_flux_normalized = normalize_spectrum(observed_flux)
    normalized_flux = normalize_spectrum(template_flux)



    z_min = np.min(wl) / np.min(rest_lines) - 1 # Take the minimum and maximum shift that it could possibly be
    z_max = np.max(wl) / np.max(rest_lines) - 1
    initial_pass = (z_min, z_max)


    best_z, scores, redshifts_final = calculate_redshift_iterative(normalized_flux, observed_flux_normalized, wl, initial_pass)

    print(f"The best redshift I found was {best_z}. The real redshift is {z_true}")

    print(f"My percent error is {abs((best_z - z_true)/z_true * 100)}.")

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [arb. units]")
    plt.title("You need the template (reference) to measure redshift!")
    plt.legend()
    plt.tight_layout()
    plt.show()


def gaussian(w, w0, amp, sigma=3.0):
    """
    Simple Gaussian line profile
    """
    return amp * np.exp(-(w - w0)**2 / (2.0 * sigma**2))



def plot_spectrum(wl, flux):
# ------------------------------------
# Plot: template vs. observed spectrum
# ------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(wl, flux, label=f"Observed spectrum (unknown $z$)", linewidth=1.2)
    

    

def normalize_spectrum(flux):
    '''
        GOAL: Remove the continuum (the smooth background light) to emphasize
        absorption & emission lines... (Not really required for this simulated data, but I read it
        was good to consider when calculating redshift of real spectra...)
    '''

    #
    N = len(flux)

    kernel_fraction = 0.01 # 1% of the total length.. 
    
    # since our simulated len of flux is 7000 it will take the median of 70 points.
    kernel_size = int(N * kernel_fraction)

    # Make sure that the window size is odd, because without it no middle
    if N % 2 == 0:
        kernel_size += 1

    # Traces the curve, or 'shape' of the Spectrum... focusing on smoothing the hills instead of the peaks.
    continum = medfilt(flux, kernel_size)

    # Subtracting the continuum leaves me with just the sharp lines. 
    flux_no_continum = flux - continum
    return (flux_no_continum - np.mean(flux_no_continum)) / np.std(flux_no_continum)




def shift_and_interpolate(wl, flux_temp, z):
    '''
        This function shifts the wavelength by an assumed redshift value and then plots the flux values based on the shift.
        My interpolate function maps any wavelength back to a flux value based on the template.
    '''
    shifted_wl = wl * (1 + z)
    interp_func = interp1d(shifted_wl, flux_temp, bounds_error=False, fill_value=0)
    return interp_func(wl)


def compute_correlate(observed, shifted_temp):
    '''
        Correlation. Described as below. the 'same' tag means that the output is the same as my input signal. Correlation is now centered. 
        returns the highest correlation point in the array.
    '''
    return np.max(correlate(observed, shifted_temp, mode='same'))

def calculate_redshift(assumed_redshifts, normalized_flux, observed_flux_normalized, wl):
    '''
        Takes every z value in the assumed redshifts, shifts the template with the rest_lines and gaussian to then cross correlate it with the observed spectrum. 
        It returns the scores (which is the cross correlation) and the redahift that has the highest cross correlation.
    '''
    scores = []
    for z in assumed_redshifts:
        shifted_temp = shift_and_interpolate(wl, normalized_flux, z)
        shifted_temp = normalize_spectrum(shifted_temp)
        score = compute_correlate(observed_flux_normalized, shifted_temp)    
        scores.append(score)
    
    return scores, assumed_redshifts[np.argmax(scores)]

def calculate_redshift_iterative(normalized_flux, observed_flux_normalized, wl, initial_pass, passes=3):
    '''
        Iteratively narrows down the best redshift.
    '''    
    # unpack the tuple
    z_min, z_max = initial_pass
    best_z = np.inf

    
    for i in range(passes):
        num_steps = 1000
        assumed_redshifts = np.linspace(z_min, z_max, num_steps)
        
        scores, best_z = calculate_redshift(assumed_redshifts, normalized_flux, observed_flux_normalized, wl)

        narrow = (z_max - z_min)/10
        z_min = max(best_z - narrow, 0)
        z_max = best_z + narrow
    
        print(f"Pass {i+1}: best_z = {best_z} in range {z_min} to {z_max}")

    return best_z, scores, assumed_redshifts

main()
