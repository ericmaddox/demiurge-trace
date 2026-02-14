from demiurge_trace import pulsar_module

from astropy.time import Time
import os

gw_time = 1187008882.4 # GW170817
is_simulation = False

print(f"Diagnosing Ensemble for GW Time: {gw_time}")
ensemble = pulsar_module.get_ensemble_data(gw_time, is_simulation)

print("-" * 30)
if not ensemble:
    print("ENSEMBLE IS EMPTY.")
else:
    print(f"ENSEMBLE CONTAINS {len(ensemble)} PULSARS:")
    for psr, (t, r) in ensemble.items():
        print(f"  - {psr}: {len(t)} points")

# Check why files might be missing from ensemble
real_data_dir = "data/real"
files = os.listdir(real_data_dir)
par_files = [f for f in files if f.endswith('.par')]
print(f"\nFiles in {real_data_dir}:")
for f in par_files:
    psr = f.replace('.par', '')
    tim = f.replace('.par', '.tim')
    has_tim = os.path.exists(os.path.join(real_data_dir, tim))
    print(f"  - {psr}: PAR exists, TIM {'exists' if has_tim else 'MISSING'}")
