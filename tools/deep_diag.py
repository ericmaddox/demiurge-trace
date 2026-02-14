from demiurge_trace import pulsar_module, ligo_module

import os
import traceback

gw_time = 1187008882.4 # GW170817
real_data_dir = "data/real"

print(f"Deep Diag for GW Time: {gw_time}")

for f in os.listdir(real_data_dir):
    if f.endswith('.par'):
        psr = f.replace('.par', '')
        print(f"\n[{psr}]")
        try:
            par_path = os.path.join(real_data_dir, f)
            tim_path = os.path.join(real_data_dir, psr + ".tim")
            
            print(f"  Attempting load_pulsar_data...")
            model, toas = pulsar_module.load_pulsar_data(par_path, tim_path)
            
            if model is None or toas is None:
                print("  FAILED: load_pulsar_data returned None")
                continue
                
            print(f"  Calculating residuals...")
            res = pulsar_module.calculate_residuals(model, toas)
            if res is None:
                print("  FAILED: calculate_residuals returned None")
                continue
                
            print(f"  SUCCESS: {len(res.time_resids)} points")
            
        except Exception as e:
            print(f"  CRASHED: {e}")
            traceback.print_exc()
