import pint.toa
import os
import numpy as np
from astropy.time import Time

gw_time_gps = 1187008882.4
gw_time_mjd = Time(gw_time_gps, format='gps').mjd
print(f"GW170817 MJD: {gw_time_mjd:.5f}")

real_data_dir = "data/real"
for f in os.listdir(real_data_dir):
    if f.endswith('.tim'):
        psr = f.replace('.tim', '')
        try:
            toas = pint.toa.get_TOAs(os.path.join(real_data_dir, f))
            mjds = toas.get_mjds().value
            diffs = np.abs(mjds - gw_time_mjd)
            min_diff_days = np.min(diffs)
            min_diff_hours = min_diff_days * 24
            print(f"[{psr}] Nearest TOA: {min_diff_days:.4f} days ({min_diff_hours:.2f} hours) away.")
        except Exception as e:
            print(f"[{psr}] Error: {e}")
