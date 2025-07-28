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
    return flux_no_continuum#(flux_no_continuum - np.mean(flux_no_continuum)) / np.std(flux_no_continuum)

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

def redshift_estimate_for_z(z_true, wl_min=4800, wl_max=100000, num_points=20000):
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
    rest_lines = np.array([5000.0, 6000.0, 6500.0])   # Emission line rest wavelengths
    amps = np.array([1.0, 0.8, 0.6])                  # Relative line strengths

    # Create a wavelength grid
    wl_linear = np.linspace(wl_min, wl_max, num_points)

    # Build template and observed flux
    template_flux_linear = build_template(wl_linear, rest_lines, amps)
    observed_flux_linear = build_observed_flux(wl_linear, rest_lines, amps, z_true)

    # Convert to logarithmic wavelength grid-- Preparing for correlation
    log_wl_grid = log_wavelength_grid(wl_min, wl_max, num_points) 
    template_flux_log = resample_to_log_wavelength(wl_linear, template_flux_linear, log_wl_grid)
    observed_flux_log = resample_to_log_wavelength(wl_linear, observed_flux_linear, log_wl_grid)

    # Normalize the spectra
    template_norm = normalize_spectrum(template_flux_log)
    observed_norm = normalize_spectrum(observed_flux_log)

    # Calculate the step size to find best shift
    delta_log_wl = log_wl_grid[1] - log_wl_grid[0]

    
    z_min = 0
    z_max = 10
    best_z = np.inf

    for iteration in range(5):
        # Converts z_min and z_max to shifts in log space
        
        max_shift = int(np.log(1 + z_max) / delta_log_wl)
        min_shift = int(np.log(1 + z_min) / delta_log_wl)

        #Switched to scipy's FFT-based correltion for speed. 
        corr = correlate(observed_norm, template_norm, mode='full')

        '''
            Create a range of shifts based on correlation length.
            Then, valid selects the shifts that correpsond to the current window
            Then filter the correlation to only include valid shifts

            (Multiplicative stretch to addition) because logarithms rule
            wl_obs = wl_rest(1+z)
            log(wl_obs)= log(wl_rest) + log(1+z)
            

            log(1+z) = best_shift x delta_log_wl
            1 + z = exp(best_shift x delta_log_wl)
            z = exp(best_shift x delta_log_wl) + 1
        '''
        # All possible lags. Even the negative (where it is not possible in cosmological redshift)
        # because the arrays need to be the same length.
        shift_range = np.arange(-len(template_norm) + 1, len(template_norm))

        # First pass: every shift including above 0 and every shift including 10.
        valid = (shift_range >= min_shift) & (shift_range <= max_shift)
        
        # indicies where valid is true to filter correlation
        corr = corr[valid]

        #get only the 'true' possible shifts 
        shift_range = shift_range[valid]

        # Find the best shift (MAX CORRELATION) 
        best_idx = np.argmax(corr)
        best_shift = shift_range[best_idx]

        #Convert best shift to redshift
        best_z = np.exp(best_shift * delta_log_wl) - 1

        print(f"Iteration {iteration}: best_shift={best_shift}, best_z={best_z:.5f}, max_corr={corr[best_idx]:.3f}")

        '''
            Taking 10 percent of the current range to narrow down the search.
            This is a simple way to narrow the search range
        '''
        narrow = (z_max - z_min) / 10
        z_min = max(best_z - narrow, 0)
        z_max = best_z + narrow

    error_percent = (best_z - z_true) / z_true * 100

    return best_z, error_percent

def main():
    error_percentages = []
    rng = np.random.default_rng()

    for i in range(1000):
        z_true = rng.uniform(0.0, 10)
        best_z, error_percent = redshift_estimate_for_z(z_true)

        error_percentages.append(error_percent)
        print(f"Run {i+1}: True z = {z_true:.5f}, Estimated z = {best_z:.5f}, Percent error = {error_percent}%\n")
    
    dis = np.linspace(1, len(error_percentages)+1, len(error_percentages))
       # Create subplots: one for percent error per run, one for histogram
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Percent error per run
    axs[0].plot(dis, error_percentages)
    axs[0].set_xlabel('Run Number')
    axs[0].set_ylabel('Percent Error')
    axs[0].set_title('Redshift Estimation Error')
    axs[0].grid(True)

    # Histogram of percent errors
    axs[1].hist(error_percentages, bins=1000, range=(0, np.max(error_percentages)), edgecolor='black')
    axs[1].set_xlabel('Percent Error')
    axs[1].set_ylabel('Number of Runs')
    axs[1].set_title('Distribution of Percent Errors')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

    # Take distribution and fit a gaussian curve to it, measure the mean and std, look at outliers
    # Focus on the outliers**
    # 