import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.signal import correlate
import matplotlib.pyplot as plt

def gaussian(w, w0, amp, sigma=3.0):
    """    
        Simple Gaussian line profile
    """
    return amp * np.exp(-(w - w0)**2 / (2.0 * sigma**2))

def max_wavelength_for_redshift(wl_max, rest_lines):
    """    
        Calculate the maximum redshift based on the maximum wavelength and rest lines.
        wl_max: Maximum wavelength in Angstroms.
        rest_lines: Array of rest wavelengths of emission lines in Angstroms.
    """
    return (wl_max / np.min(rest_lines)) - 1

def normalize_spectrum(flux):
    """    
        Normalize the spectrum by removing the continuum and standardizing.
    """

    N = len(flux)

    kernel_fraction = 0.01 # 1% of the total length.. 
    
    # since our simulated len of flux is 7000 it will take the median of 70 points.
    kernel_size = int(N * kernel_fraction)

    # Make sure that the window size is odd, because without it no middle
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Traces the curve, or 'shape' of the Spectrum... focusing on smoothing the hills instead of the peaks.
    continuum = medfilt(flux, kernel_size)

    # Subtracting the continuum leaves me with just the sharp lines. 
    flux_no_continuum = flux - continuum
    return (flux_no_continuum - np.mean(flux_no_continuum)) / np.std(flux_no_continuum)

def log_wavelength_grid(wl_min, wl_max, num_points):
    """ 
        Creates a logarithmic wavelength grid.
    """
    return np.linspace(np.log(wl_min), np.log(wl_max), num_points)

def resample_to_log_wavelength(wl_linear, flux_linear, log_wl_grid):
    """    Resample the flux values to a logarithmic wavelength grid.
        wl_linear: Linear wavelength grid.
        flux_linear: Corresponding flux values.
        log_wl_grid: Logarithmic wavelength grid to resample to.
    """
    interp_func = interp1d(np.log(wl_linear), flux_linear, bounds_error=False, fill_value=0)
    return interp_func(log_wl_grid)

def build_template(wl_linear, rest_lines, amps):
    '''
        Builds a template spectrum on a linear wavelength grid.
        rest_lines: Wavelengths of the emission lines in Angstroms.
        amps: Amplitudes of the emission lines.
    '''
    template_flux = np.zeros_like(wl_linear)
    for w0, amp in zip(rest_lines, amps):
        template_flux += gaussian(wl_linear, w0, amp)
    return template_flux

def build_observed_flux(wl_linear, rest_lines, amps, z_true):
    """    
        Builds the observed flux spectrum based on the true redshift.
    """
    shifted_lines = rest_lines * (1 + z_true)
    observed_flux = np.zeros_like(wl_linear)
    for w0, amp in zip(shifted_lines, amps):
        observed_flux += gaussian(wl_linear, w0, amp)
    rng = np.random.default_rng(42)
    observed_flux += 0.05 * rng.normal(size=observed_flux.size) # add noise
    return observed_flux


def redshift_estimate_for_z(flux, wl, templates, z_min=0, z_max=10, num_points=20000, iterations=5):
    """ 
        Estimate the redshift for a given true redshift value.
        wl_min: Minimum wavelength in Angstroms.
        wl_max: Maximum wavelength in Angstroms.
        num_points: Number of points in the wavelength grid.

        Improvements:
        Instead of spacing linearly, I have opted for a logarithmic grid.
        This is because (linearly) redshift is multiplicative.. but in log space, it becomes additive.
        Instead of a (stretch) on the spectrum.. it just becomes a shift.
        So, a redshift of 1.0 becomes log(2.0) = 0.693, 
        and a redshift of 10.0 becomes log(11.0) = 2.398.
        Instead of (shift_and_interpolate) every time, (which is what was taking up most of the time)
        Instead I only have to slide it accross the spectrum once. Better than reshaping each time. 
        O(n) vs O(n^2)! Wayyy faster! better for bigger redshifts too. 


        Now it can handle larger redshifts :)) (because it's less costly to calculate)
        (and it's really fast I don't have to worry about it running slow on my little laptop)
    """
    # Convert observed spectrum to log space
    log_wl_grid = log_wavelength_grid(np.min(wl), np.max(wl), num_points)
    observed_flux_log = resample_to_log_wavelength(wl, flux, log_wl_grid)

    # Don't normalize — instead subtract mean to remove DC offset for correlation
    observed_flux_log -= np.mean(observed_flux_log)

    delta_log_wl = log_wl_grid[1] - log_wl_grid[0]

    best_corr_value = -np.inf
    best_z = None
    best_template_index = -1

    for idx, (template_wl, template_flux) in enumerate(templates):
        # Resample the template to the same log grid
        template_flux_log = resample_to_log_wavelength(template_wl, template_flux, log_wl_grid)
        template_flux_log -= np.mean(template_flux_log)  # Remove DC offset

        zmin = z_min
        zmax = z_max

        for _ in range(iterations):
            max_shift = int(np.log(1 + zmax) / delta_log_wl)
            min_shift = int(np.log(1 + zmin) / delta_log_wl)
            '''
                Create a range of shifts based on correlation length.
                Then, valid selects the shifts that correpsond to the current window
                Then filter the correlation to only include valid shifts
            '''
            # all lags from 0 to pos because redshift 
            shift_range = np.arange(-len(template_flux_log) + 1, len(template_flux_log))
            # indicies where valid is true to filter correlation
            valid = (shift_range >= min_shift) & (shift_range <= max_shift)

            corr = correlate(observed_flux_log, template_flux_log, mode='full')
            corr = corr[valid]
            shift_range = shift_range[valid]

            # Find the best shift (MAX CORRELATION) (Didnt really change from simulation one)
            best_idx = np.argmax(corr)
            shift = shift_range[best_idx]
            z = np.exp(shift * delta_log_wl) - 1
            corr_val = corr[best_idx]

            '''
                Taking 10 percent of the current range to narrow down the search.
                This is a simple way to narrow the search range
            '''
            window = (zmax - zmin) / 10
            zmin = max(z - window, 0)
            zmax = z + window
        # Update
        if corr_val > best_corr_value:
            best_corr_value = corr_val
            best_z = z
            best_template_index = idx

        print(f"Template {idx} best_z: {z:.5f}, max_corr: {corr_val:.3f}")

    return best_z, best_template_index