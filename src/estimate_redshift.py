#!/usr/bin/env python3
import argparse
import sys
import os

# Add src directory to import path
this_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(this_dir, "src"))
sys.path.append(src_path)

import redshift_cross_correlation as mrc
import redshift_cc_sdss as crs

def main():
    parser = argparse.ArgumentParser(description="Estimate redshift using cross-correlation with SDSS spectrum.")
    parser.add_argument("--plate", type=int, required=True, help="SDSS plate number")
    parser.add_argument("--mjd", type=int, required=True, help="SDSS MJD")
    parser.add_argument("--fiberID", type=int, required=True, help="SDSS fiber ID")
    parser.add_argument("--type", type=str, default="ALL", choices=["GALAXY", "QSO", "STAR", "ALL"],
                        help="Template type to use for matching (default: ALL)")
    parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift to consider (default: 0.0)")
    parser.add_argument("--zmax", type=float, default=7.0, help="Maximum redshift to consider (default: 7.0)")

    args = parser.parse_args()

    print("Downloading SDSS spectrum...")
    spectrum = crs.get_spectrum(plate=args.plate, mjd=args.mjd, fiberID=args.fiberID)

    flux, wavelength = spectrum

    print(f"Loading template spectra: type = {args.type}")
    templates = crs.get_template_spectra(args.type.upper())

    print("Performing cross-correlation...")
    template_idx, best_z, best_score = mrc.cross_correlate_redshift(
        wavelength, flux, templates, z_min=args.zmin, z_max=args.zmax
    )

    print("\nRedshift Estimation Complete:")
    print(f"   Template index: {template_idx:03d}")
    print(f"   Estimated redshift (z): {best_z:.5f}")
    print(f"   Correlation score: {best_score:.3f}")

if __name__ == "__main__":
    main()
