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
        baseline_residuals = pulsar_residuals
        
    baseline_rms = np.sqrt(np.mean(baseline_residuals**2))
    
    # Calculate Window RMS
    window_rms = np.sqrt(np.mean(window_residuals**2))
    
    # Calculate Sigma deviation
    if baseline_rms > 0:
        baseline_std = np.std(baseline_residuals)
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


# ---------------------------------------------------------------------------
# Monte Carlo Significance Test
# ---------------------------------------------------------------------------

def monte_carlo_test(gw_time, pulsar_times, pulsar_residuals,
                     window_seconds=10.0, n_iterations=10000):
    """
    Compute a p-value for the observed window sigma by shuffling residuals.

    Procedure:
      1. Compute the observed sigma for the real window placement.
      2. Randomly shuffle the residual values (breaking the time association)
         and recompute the window sigma for each trial.
      3. p-value = fraction of trials where shuffled sigma >= observed sigma.

    A low p-value (< 0.05) means the observed deviation is unlikely due to
    chance alone.

    Returns:
        dict with keys: observed_sigma, p_value, n_iterations, n_exceed
    """
    observed = audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds)
    obs_sigma = observed["sigma"]

    rng = np.random.default_rng(42)
    n_exceed = 0

    mask = (pulsar_times >= gw_time - window_seconds) & (pulsar_times <= gw_time + window_seconds)
    n_in_window = int(np.sum(mask))

    if n_in_window == 0 or len(pulsar_residuals) < 2:
        return {
            "observed_sigma": obs_sigma,
            "p_value": 1.0,
            "n_iterations": n_iterations,
            "n_exceed": 0,
        }

    baseline_idx = ~mask
    for _ in range(n_iterations):
        shuffled = rng.permutation(pulsar_residuals)
        w_rms = np.sqrt(np.mean(shuffled[mask] ** 2))
        b_res = shuffled[baseline_idx]
        b_rms = np.sqrt(np.mean(b_res ** 2))
        b_std = np.std(b_res)
        trial_sigma = (w_rms - b_rms) / b_std if b_std > 0 else 0
        if trial_sigma >= obs_sigma:
            n_exceed += 1

    p_value = n_exceed / n_iterations

    return {
        "observed_sigma": obs_sigma,
        "p_value": p_value,
        "n_iterations": n_iterations,
        "n_exceed": n_exceed,
    }


# ---------------------------------------------------------------------------
# Sliding Window Sweep
# ---------------------------------------------------------------------------

def sliding_window_sweep(gw_time, pulsar_times, pulsar_residuals,
                         window_seconds=10.0, n_offsets=200):
    """
    Sweep an analysis window across the timeline and compute sigma at each offset.

    The sweep covers the full span of pulsar observations. At each offset
    position the window is centered at (gw_time + offset) and sigma is
    computed identically to audit_lag.

    Returns:
        dict with keys:
          offsets   — 1-D array of time offsets relative to gw_time (seconds)
          sigmas    — 1-D array of sigma values at each offset
          gw_sigma  — the sigma at offset = 0 (the actual event)
    """
    t_min = float(np.min(pulsar_times))
    t_max = float(np.max(pulsar_times))

    # Generate offsets spanning full observation span
    centers = np.linspace(t_min + window_seconds, t_max - window_seconds, n_offsets)
    offsets = centers - gw_time

    sigmas = np.empty(n_offsets)
    for i, center in enumerate(centers):
        mask = (pulsar_times >= center - window_seconds) & (pulsar_times <= center + window_seconds)
        w_res = pulsar_residuals[mask]
        b_res = pulsar_residuals[~mask]

        if len(w_res) == 0 or len(b_res) < 2:
            sigmas[i] = 0.0
            continue

        w_rms = np.sqrt(np.mean(w_res ** 2))
        b_rms = np.sqrt(np.mean(b_res ** 2))
        b_std = np.std(b_res)
        sigmas[i] = (w_rms - b_rms) / b_std if b_std > 0 else 0.0

    # Sigma at exact event time
    gw_result = audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds)

    return {
        "offsets": offsets,
        "sigmas": sigmas,
        "gw_sigma": gw_result["sigma"],
    }


# ---------------------------------------------------------------------------
# Ensemble Audit (with Monte Carlo p-values)
# ---------------------------------------------------------------------------

def audit_ensemble(gw_time, ensemble_data, window_seconds=10.0,
                   monte_carlo=True, mc_iterations=10000):
    """
    Performs a coherent audit across an ensemble of pulsars.
    Looks for synchronized lag artifacts.
    Optionally runs Monte Carlo significance tests per pulsar.
    """
    results = {}
    all_sigmas = []
    
    for psr_name, (times, residuals) in ensemble_data.items():
        res = audit_lag(gw_time, times, residuals, window_seconds)

        # Add Monte Carlo p-value
        if monte_carlo:
            mc = monte_carlo_test(gw_time, times, residuals,
                                  window_seconds, n_iterations=mc_iterations)
            res["p_value"] = mc["p_value"]
            res["mc_n_exceed"] = mc["n_exceed"]
            res["mc_iterations"] = mc["n_iterations"]
        else:
            res["p_value"] = None

        results[psr_name] = res
        if res['window_rms'] > 0:
            all_sigmas.append(res['sigma'])
            
    n_pulsars = len(all_sigmas)
    
    ensemble_sigma = np.mean(all_sigmas) if n_pulsars > 0 else 0
    n_detectors_above_1s = sum(1 for r in results.values() if r.get('sigma', 0) > 1.0)
    is_ensemble_artifact = (ensemble_sigma > 3.0) or (n_detectors_above_1s >= 3 and n_pulsars >= 5)
    
    return {
        "ensemble_sigma": ensemble_sigma,
        "n_pulsars_in_window": n_pulsars,
        "n_detectors_above_1s": n_detectors_above_1s,
        "is_ensemble_artifact": is_ensemble_artifact,
        "pulsar_results": results,
        "gw_time": gw_time
    }
