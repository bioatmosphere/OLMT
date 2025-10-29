#!/usr/bin/env python
"""
Helper script to convert legacy parameter files to lognormal priors.

Usage:
    python convert_to_lognormal.py input_parm_list output_parm_list [log_std]

Arguments:
    input_parm_list: Path to input parameter file (legacy 4-column format)
    output_parm_list: Path to output parameter file (with lognormal priors)
    log_std: (optional) Default log_std value (default: 0.5)
"""

import numpy as np
import sys
import os

def convert_to_lognormal(input_file, output_file, default_log_std=0.5):
    """
    Convert a legacy parameter file to use lognormal priors.

    Parameters
    ----------
    input_file : str
        Path to input parameter file (format: parm_name pft min max)
    output_file : str
        Path to output parameter file (format: parm_name pft lognormal log_mean log_std min max)
    default_log_std : float
        Default value for log_std (controls spread of distribution)
    """

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return False

    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Process and write output
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Parameter file converted to lognormal priors\n")
        f.write(f"# Generated from: {input_file}\n")
        f.write(f"# Default log_std: {default_log_std}\n")
        f.write("#\n")
        f.write("# Format: parameter_name pft lognormal log_mean log_std min max\n")
        f.write("#\n")
        f.write("# Lognormal parameters:\n")
        f.write("# - log_mean = ln(geometric_mean) = (ln(min) + ln(max)) / 2\n")
        f.write("# - log_std controls spread:\n")
        f.write("#   * 0.3 = narrow (68% within [0.74, 1.35] × median)\n")
        f.write("#   * 0.5 = moderate (68% within [0.61, 1.65] × median)\n")
        f.write("#   * 0.7 = wide (68% within [0.50, 2.01] × median)\n")
        f.write("#\n\n")

        # Process each line
        for line in lines:
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                f.write(line)
                continue

            # Remove inline comments
            if '#' in line:
                line_data = line[:line.index('#')]
                comment = line[line.index('#'):]
            else:
                line_data = line
                comment = ''

            vals = line_data.split()

            if len(vals) < 4:
                # Malformed line, write as-is
                f.write(line)
                continue

            parm_name = vals[0]
            pft = vals[1]
            min_val = float(vals[2])
            max_val = float(vals[3])

            # Calculate lognormal parameters
            if min_val <= 0 or max_val <= 0:
                print(f"Warning: Parameter '{parm_name}' has non-positive bounds [{min_val}, {max_val}]")
                print(f"         Lognormal requires positive values. Keeping as uniform distribution.")
                f.write(f"{parm_name} {pft} uniform {min_val} {max_val}  # Non-positive bounds\n")
                continue

            # Geometric mean
            geom_mean = np.sqrt(min_val * max_val)
            log_mean = np.log(geom_mean)

            # Adjust log_std based on parameter range
            range_ratio = max_val / min_val
            if range_ratio > 50:
                # Very wide range
                log_std = 0.7
            elif range_ratio > 10:
                # Wide range
                log_std = 0.6
            elif range_ratio < 3:
                # Narrow range
                log_std = 0.3
            else:
                # Moderate range
                log_std = default_log_std

            # Write converted line
            f.write(f"# Original: {parm_name} {pft} {min_val} {max_val}  "
                   f"(geom_mean={geom_mean:.3f}, ratio={range_ratio:.1f})\n")
            f.write(f"{parm_name} {pft} lognormal {log_mean:.3f} {log_std} {min_val} {max_val}")
            if comment:
                f.write(f"  {comment}")
            else:
                f.write("\n")
            f.write("\n")

    print(f"✓ Conversion complete!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Parameters converted: {len([l for l in lines if l.strip() and not l.strip().startswith('#')])}")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    log_std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    success = convert_to_lognormal(input_file, output_file, log_std)
    sys.exit(0 if success else 1)
