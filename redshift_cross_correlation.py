import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def gaussian(w, w0, amp, sigma=3.0):
    """
        Simple Gaussian line profile.
        w0 = center wavelength.
        amp = peak intensity.
        sigma = width of the Gaussian.
    """
    return amp * np.exp(-(w - w0) ** 2 / (2.0 * sigma ** 2))


def normalize_spectrum(flux):
    '''
        GOAL: Remove the continuum (the smooth background light) to emphasize
        absorption & emission lines.

        This uses a median filter to estimate the general 'shape' or hills of the spectrum,
        subtracts that from the original flux to isolate sharp features,
        then normalizes to zero-mean, unit-variance.
    '''
    N = len(flux)
    kernel_fraction = 0.01  # 1% of the data length
    kernel_size = int(N * kernel_fraction)

    if kernel_size % 2 == 0:
        kernel_size += 1  # Make sure it's odd to ensure a middle point

    continuum = medfilt(flux, kernel_size)
    flux_no_continuum = flux - continuum

    return (flux_no_continuum - np.mean(flux_no_continuum)) / np.std(flux_no_continuum)


def log_wavelength_grid(wl_min, wl_max, num_points):
    '''
        Creates a logarithmically spaced wavelength grid.
        This is essential for detecting redshift via shifting in log-wavelength space.
    '''
    return np.linspace(np.log(wl_min), np.log(wl_max), num_points)


def resample_to_log_wavelength(wl_linear, flux_linear, log_wl_grid):
    '''
        Interpolates a linear wavelength spectrum onto a log-wavelength grid.
    '''
    interp_func = interp1d(np.log(wl_linear), flux_linear, bounds_error=False, fill_value=0)
    return interp_func(log_wl_grid)


def build_template_on_linear_grid(wl_linear, rest_lines, amps):
    '''
        Constructs a template spectrum in LINEAR wavelength space using Gaussian lines.
        Adds weak continuum and "ghost" lines at high redshifts to assist detection.
    '''
    template_flux = 0.02 + np.zeros_like(wl_linear)  # Weak continuum baseline

    for w0, amp in zip(rest_lines, amps):
        template_flux += gaussian(wl_linear, w0, amp)

        # Extra "ghost" lines at high-z positions to help prevent detection failure
        for z_pad in [5, 10, 15]:
            template_flux += gaussian(wl_linear, w0 * (1 + z_pad), amp * 0.5)

    return template_flux


def build_observed_flux(wl_linear, rest_lines, amps, z_true):
    '''
        Applies the unknown "true" redshift to generate the observed flux (our "data").
        Adds a weak continuum to simulate realistic conditions.
    '''
    shifted_lines = rest_lines * (1 + z_true)
    observed_flux = 0.02 + np.zeros_like(wl_linear)
    
    for w0, amp in zip(shifted_lines, amps):
        observed_flux += gaussian(wl_linear, w0, amp)
    
    return observed_flux


def redshift_estimate_for_z_true(z_true, wl_min=4800, wl_max=80000, num_points=1000):
    '''
        Attempts to recover the unknown redshift using cross-correlation over log-wavelength space.
        Uses shifting of the template spectrum to find the best alignment.
    '''
    z_max_detectable = 10  # Ensure our wavelength range can handle maximum expected redshift
    wl_max = max(wl_max, wl_min * (1 + z_max_detectable) * 1.1)

    rest_lines = np.array([5000.0, 6000.0, 6500.0])  # H‑beta, [O III], H‑alpha-ish placeholders
    amps = np.array([1.0, 0.8, 0.6])                 # Relative line strengths

    wl_linear = np.linspace(wl_min, wl_max, num_points)

    # Build clean template & observed flux (with noise)
    template_flux_linear = build_template_on_linear_grid(wl_linear, rest_lines, amps)
    observed_flux_linear = build_observed_flux(wl_linear, rest_lines, amps, z_true)

    rng = np.random.default_rng(42)
    observed_flux_linear += 0.05 * rng.normal(size=observed_flux_linear.size)

    # Project both onto log-wavelength grid
    log_wl_grid = log_wavelength_grid(wl_min, wl_max, num_points)
    template_flux_log = resample_to_log_wavelength(wl_linear, template_flux_linear, log_wl_grid)
    observed_flux_log = resample_to_log_wavelength(wl_linear, observed_flux_linear, log_wl_grid)

    # Normalize spectra for proper comparison
    template_norm = normalize_spectrum(template_flux_log)
    observed_norm = normalize_spectrum(observed_flux_log)

    delta_log_wl = log_wl_grid[1] - log_wl_grid[0]

    z_min, z_max = 0, 10
    best_z = None

    # 5 iterations of narrowing down best shift
    for iteration in range(5):
        max_shift = int(np.log(1 + z_max) / delta_log_wl)
        min_shift = int(np.log(1 + z_min) / delta_log_wl)
        shifts = np.arange(min_shift, max_shift + 1)

        scores = []

        # Pad template to prevent edge effects during shifting
        padded_template = np.pad(template_norm, pad_width=template_norm.size, mode='constant', constant_values=0)

        for shift in shifts:
            shifted_template = np.roll(padded_template, shift)[template_norm.size : 2 * template_norm.size]
            score = np.sum(observed_norm * shifted_template)  # Simple dot product as correlation measure
            scores.append(score)

        best_shift = shifts[np.argmax(scores)]
        best_z = np.exp(best_shift * delta_log_wl) - 1

        # Narrow search range for next iteration
        safety_margin = 0.1
        narrow = max((z_max - z_min) / 10, safety_margin)
        z_min = max(best_z - narrow, 0)
        z_max = best_z + narrow

    error_percent = abs((best_z - z_true) / z_true) * 100 if z_true != 0 else 0
    
    plt.plot(wl_linear, observed_norm)
    plt.plot(wl_linear, template_norm)
    plt.show()

'''
def main():
    rng = np.random.default_rng()
    #error_p = []
    #disti = np.linspace(0, 1000, 10)
    
    z_true = rng.uniform(0.0, 10.0)
    best_z, error_percent,obs, temp = redshift_estimate_for_z_true(z_true)
    print(f"Run {i+1}: True z = {z_true:.5f}, Estimated z = {best_z:.5f}, Percent error = {error_percent:.2f}%")

    plt.plot()
    #error_p.append(error_percent)
    #plt.plot(disti, error_p)
    


if __name__ == "__main__":
    main()
'''
rng = np.random.default_rng()
redshift_estimate_for_z_true(rng.uniform(0.0, 10.0))