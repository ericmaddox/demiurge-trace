import pint.toa
import os
from astropy.time import Time

# GW170817 MJD
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
            print(f"[{psr}] MJD Range: {min(mjds):.2f} to {max(mjds):.2f} (Count: {len(mjds)})")
            if min(mjds) <= gw_time_mjd <= max(mjds):
                print(f"  --> OK: Event falls within range.")
            else:
                print(f"  --> OUTSIDE: Event is not in range.")
        except Exception as e:
            print(f"[{psr}] Error: {e}")
