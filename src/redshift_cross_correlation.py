from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

def make_adaptive_z_grid(z_min=0.0, z_max=10.0):
    '''
        Create a redshift grid that adjusts its resolution for better precision:
        Fine resolution (0.0005) for z < 0.2
        Moderate resolution (0.002) for 0.2 ≤ z < 1.0
        Coarser (0.01) for z ≥ 1.0    
    '''
    low_z = np.arange(z_min, 0.2, 0.0005)
    mid_z = np.arange(0.2, 1.0, 0.002)
    high_z = np.arange(1.0, z_max, 0.01)
    return np.concatenate([low_z, mid_z, high_z])

def log_wavelength_grid(wl_min, wl_max, num_points=30000):
    '''
        Generate a wavelength grid that is logarithmically spaced
        Because redshift (z) stretches wavelengths multiplicatively: λ_obs = λ_emit × (1 + z)
    '''
    log_min = np.log10(wl_min)
    log_max = np.log10(wl_max)
    return np.logspace(log_min, log_max, num=num_points)


def high_pass(flux, sigma=100):
    '''
        Subtracts a smoothed version of the flux to remove broad continuum features (high-pass)
        Useful for focusing on narrow absorption/emission features
    '''
    smoothed = gaussian_filter1d(flux, sigma=sigma, mode='nearest')
    return flux - smoothed

def normalize_segment(flux):
    '''
        Normalize a flux segment to have zero mean and unit standard deviation
        Ensures amplitude does not affect the cross-correlation score (only shape matters)
    '''
    flux = flux - np.nanmean(flux)
    std = np.nanstd(flux)
    return flux / std if std > 0 else flux

# Main function to estimate redshift by cross-correlating observed spectra with template spectra
def cross_correlate_redshift(observed_wavelength, observed_flux, template_spectra, z_min=0.0, z_max=10.0):
    best_template_index = -1
    best_z = 0.0
    best_score = -np.inf

    # Determine how far red in λ we need to go, based on maximum redshift
    template_max_wl = max(t[0][-1] for t in template_spectra)
    wavelength_max_needed = template_max_wl * (1 + z_max) * 1.05  # 5% extra margin
    grid_wavelength = log_wavelength_grid(3500, wavelength_max_needed, num_points=30000)

    # Resample the observed spectrum onto the log-wavelength grid
    obs_interp = interp1d(observed_wavelength, observed_flux, kind='linear',
                          bounds_error=False, fill_value=np.nan)
    obs_resampled = obs_interp(grid_wavelength)
    obs_resampled_hp = high_pass(obs_resampled)

    for i, (template_wavelength, template_flux) in enumerate(template_spectra):
        z_grid = make_adaptive_z_grid(z_min, z_max)

        for z in z_grid:
            # Apply redshift to the template: λ_obs = λ_emit × (1 + z)
            shifted_wavelength = template_wavelength * (1 + z)

            # Resample the shifted template onto the same log grid
            shifted_interp = interp1d(shifted_wavelength, template_flux, kind='linear',
                                      bounds_error=False, fill_value=np.nan)
            template_resampled = shifted_interp(grid_wavelength)
            template_resampled_hp = high_pass(template_resampled)

            # Mask invalid (NaN) regions for fair correlation
            valid = (~np.isnan(obs_resampled_hp)) & (~np.isnan(template_resampled_hp))
            if np.sum(valid) < 100:  # Skip poorly overlapping templates
                continue

            # Normalize both the raw and high-pass spectra to focus only on relative shape
            obs_seg = normalize_segment(obs_resampled[valid])
            temp_seg = normalize_segment(template_resampled[valid])
            obs_hp = normalize_segment(obs_resampled_hp[valid])
            temp_hp = normalize_segment(template_resampled_hp[valid])

            # Final correlation score:
            # A blend of shape-based match (raw) and feature-based match (high-pass)
            # This gives a balanced score between continuum shape and fine structure
            score = 0.5 * np.sum(obs_seg * temp_seg) + 0.5 * np.sum(obs_hp * temp_hp)

            if score > best_score:
                best_score = score
                best_z = z
                best_template_index = i

    print(f"\n>> Best overall match: Template {best_template_index:03d} with z = {best_z:.5f}, score = {best_score:.3f}")
    plt.figure(figsize=(20,4))
    plt.plot(observed_wavelength, observed_flux)
    plt.plot(template_spectra[best_template_index][0], template_spectra[best_template_index][1])
    return best_template_index, best_z, best_score
