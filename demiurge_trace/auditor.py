import numpy as np

def audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds=10.0):
    """
    Audits pulsar residuals for lag spikes around a GW event.
    
    Args:
        gw_time (float): GPS timestamp of the GW event.
        pulsar_times (array): Array of pulsar observation times (GPS seconds).
        pulsar_residuals (array): Array of pulsar timing residuals (seconds).
        window_seconds (float): Window size to check for lag (seconds).
        
    Returns:
        dict: Analysis results.
    """
    
    # Define window mask
    mask = (pulsar_times >= gw_time - window_seconds) & (pulsar_times <= gw_time + window_seconds)
    
    window_residuals = pulsar_residuals[mask]
    
    if len(window_residuals) == 0:
        return {
            "window_rms": 0,
            "baseline_rms": 0,
            "sigma": 0,
            "is_artifact": False,
            "gw_time": gw_time,
            "message": "No pulsar data in window"
        }

    # Calculate Baseline RMS (using data outside the window)
    baseline_residuals = pulsar_residuals[~mask]
    if len(baseline_residuals) == 0:
         # Fallback if window covers everything (unlikely)
        baseline_residuals = pulsar_residuals
        
    baseline_rms = np.sqrt(np.mean(baseline_residuals**2))
    
    # Calculate Window RMS
    window_rms = np.sqrt(np.mean(window_residuals**2))
    
    # Calculate Sigma deviation
    if baseline_rms > 0:
        sigma = (window_rms - baseline_rms) / baseline_rms # Relative deviation? 
        # Or standard z-score of the RMS? 
        # Usually we compare the *level* of RMS. 
        # Let's say: how many times the baseline RMS is the window RMS?
        # A 3-sigma event usually means the *individual residuals* are 3-sigma away.
        # But here we are looking for a *cluster* of residuals that increase the RMS.
        # Let's use ratio for now and flag if it's > 3x (arbitrary but fits "3 sigma" loosely in colloquium)
        # Better: (Window RMS - Baseline Mean) / Baseline StdDev .. but we only have RMS.
        
        # Let's stick to the prompt's logic: "If specific window RMS is > 3.0 sigma of the baseline"
        # This implies we treat the baseline residuals as a distribution.
        baseline_std = np.std(baseline_residuals)
        
        # We are comparing the *RMS* of the window to the *distribution* of the baseline.
        # If the window RMS is significantly higher than the baseline RMS.
        
        sigma_deviation = (window_rms - baseline_rms) / baseline_std if baseline_std > 0 else 0
    else:
        sigma_deviation = 0

    is_artifact = sigma_deviation > 3.0
    
    return {
        "window_rms": window_rms,
        "baseline_rms": baseline_rms,
        "sigma": sigma_deviation,
        "is_artifact": is_artifact,
        "gw_time": gw_time
    }


def audit_ensemble(gw_time, ensemble_data, window_seconds=10.0):
    """
    Performs a coherent audit across an ensemble of pulsars.
    Looks for synchronized lag artifacts.
    """
    results = {}
    all_sigmas = []
    
    for psr_name, (times, residuals) in ensemble_data.items():
        res = audit_lag(gw_time, times, residuals, window_seconds)
        results[psr_name] = res
        if res['window_rms'] > 0:
            all_sigmas.append(res['sigma'])
            
    n_pulsars = len(all_sigmas)
    
    # Calculate Coherent Ensemble Score (average sigma)
    ensemble_sigma = np.mean(all_sigmas) if n_pulsars > 0 else 0
    
    # Count how many pulsars saw a > 1 sigma deviation concurrently
    n_detectors_above_1s = sum(1 for r in results.values() if r.get('sigma', 0) > 1.0)
    
    # Flag as artifact if ensemble sigma is high or if many pulsars glitch together
    is_ensemble_artifact = (ensemble_sigma > 3.0) or (n_detectors_above_1s >= 3 and n_pulsars >= 5)
    
    return {
        "ensemble_sigma": ensemble_sigma,
        "n_pulsars_in_window": n_pulsars,
        "n_detectors_above_1s": n_detectors_above_1s,
        "is_ensemble_artifact": is_ensemble_artifact,
        "pulsar_results": results,
        "gw_time": gw_time
    }

