import numpy as np
import pint.models
import pint.toa
import astropy.units as u
from astropy.time import Time

def load_pulsar_data(par_file, tim_file):
    """
    Loads pulsar timing data from .par and .tim files.
    """
    try:
        model = pint.models.get_model(par_file)
        toas = pint.toa.get_TOAs(tim_file, planets=True)
        return model, toas

    except Exception as e:
        print(f"Error loading pulsar data: {e}")
        return None, None


def calculate_residuals(model, toas):
    """
    Calculates timing residuals.
    """
    try:
        import pint.residuals
        res = pint.residuals.Residuals(toas, model)
        return res
    except Exception as e:
        print(f"Error calculating residuals: {e}")
        return None



def generate_mock_pulsar_data(event_gps_time, is_simulation=False):
    """
    Generates mock residuals for testing.
    If is_simulation is True, adds a 'lag spike' at the event time.
    """
    # Create a time range around the event
    start_time = event_gps_time - 100
    end_time = event_gps_time + 100
    times = np.linspace(start_time, end_time, 1000)
    
    # Baseline noise (white noise) ~ 1 microsecond RMS
    noise_level = 1e-6 
    residuals = np.random.normal(0, noise_level, len(times))
    
    if is_simulation:
        # Inject "Lag Spike" - a Gaussian smeared delay
        # Center at event time, width 1s, amplitude 10 microseconds
        lag_amplitude = 10e-6
        lag_width = 1.0
        lag_spike = lag_amplitude * np.exp(-0.5 * ((times - event_gps_time) / lag_width)**2)
        residuals += lag_spike
        
    return times, residuals

def get_pulsar_data(pulsar_name, event_gps_time, is_simulation=False):
    """
    High-level function to get pulsar data.
    Checks for real data files in data/real/ first.
    If not found, generates mock data.
    """
    import os
    
    real_data_dir = "data/real"
    par_file = os.path.join(real_data_dir, f"{pulsar_name}.par")
    tim_file = os.path.join(real_data_dir, f"{pulsar_name}.tim")
    
    if os.path.exists(par_file) and os.path.exists(tim_file):
        print(f"Found REAL data for {pulsar_name} in {real_data_dir}")
        model, toas = load_pulsar_data(par_file, tim_file)
        if model and toas:
            res = calculate_residuals(model, toas)
            
            if res is not None:
                residuals = res.time_resids.to(u.s).value
                
                # Convert TOA MJDs (Quantity) to Astropy Time then to GPS
                # Use res.toas to ensure alignment with residuals
                mjds = res.toas.get_mjds()
                t_obj = Time(mjds, format='mjd')
                gps_times = t_obj.gps
                
                # Ensure lengths match
                if len(gps_times) != len(residuals):
                    print(f"WARNING: Length mismatch! GPS: {len(gps_times)}, Res: {len(residuals)}")
                    min_len = min(len(gps_times), len(residuals))
                    gps_times = gps_times[:min_len]
                    residuals = residuals[:min_len]
                
                # If simulation requested on REAL data, we inject the signal into the residuals.
                if is_simulation:
                    print("Injecting SIMULATED LAG into REAL data...")
                    lag_amplitude = 10e-6
                    lag_width = 1.0
                    lag_spike = lag_amplitude * np.exp(-0.5 * ((gps_times - event_gps_time) / lag_width)**2)
                    residuals += lag_spike
                    
                return gps_times, residuals
            else:
                return np.array([]), np.array([])
    
    print(f"Real data not found for {pulsar_name} in {real_data_dir}")
    print("Please run 'python setup_data.py' to download the NANOGrav 15-year data set.")
    raise FileNotFoundError(f"Real data files for {pulsar_name} not found.")


def get_ensemble_data(gw_time, is_simulation=False):
    """
    Loads data for all available pulsars in data/real.
    Returns: dict {pulsar_name: (gps_times, residuals)}
    """
    import os
    real_data_dir = "data/real"
    ensemble = {}
    
    # Find all pulsars with both .par and .tim files
    if not os.path.exists(real_data_dir):
        return {}
        
    files = os.listdir(real_data_dir)
    par_files = [f for f in files if f.endswith('.par')]
    
    for par_name in par_files:
        psr_name = par_name.replace('.par', '')
        tim_path = os.path.join(real_data_dir, f"{psr_name}.tim")
        
        if os.path.exists(tim_path):
            print(f"Loading ensemble member: {psr_name}...")
            try:
                times, residuals = get_pulsar_data(psr_name, gw_time, is_simulation)
                if len(times) > 0:
                    ensemble[psr_name] = (times, residuals)
            except Exception as e:
                print(f"  [SKIPPED] {psr_name}: {e}")
                
    return ensemble

