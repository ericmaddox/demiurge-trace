import numpy as np


def audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds=10.0):
    """
    Audits pulsar residuals for lag spikes around a GW event.
    """
    mask = (pulsar_times >= gw_time - window_seconds) & (pulsar_times <= gw_time + window_seconds)
    window_residuals = pulsar_residuals[mask]

    if len(window_residuals) == 0:
        return {
            "window_rms": 0, "baseline_rms": 0, "sigma": 0,
            "is_artifact": False, "gw_time": gw_time, "message": "No pulsar data in window"
        }

    baseline_residuals = pulsar_residuals[~mask]
    if len(baseline_residuals) == 0:
        baseline_residuals = pulsar_residuals

    baseline_rms = np.sqrt(np.mean(baseline_residuals**2))
    window_rms = np.sqrt(np.mean(window_residuals**2))

    if baseline_rms > 0:
        baseline_std = np.std(baseline_residuals)
        sigma_deviation = (window_rms - baseline_rms) / baseline_std if baseline_std > 0 else 0
    else:
        sigma_deviation = 0

    return {
        "window_rms": window_rms, "baseline_rms": baseline_rms,
        "sigma": sigma_deviation, "is_artifact": sigma_deviation > 3.0,
        "gw_time": gw_time
    }


# ---------------------------------------------------------------------------
# Monte Carlo Significance Test
# ---------------------------------------------------------------------------

def monte_carlo_test(gw_time, pulsar_times, pulsar_residuals,
                     window_seconds=10.0, n_iterations=10000):
    """
    Compute a p-value by shuffling residuals n_iterations times.
    p-value = fraction of trials where shuffled sigma >= observed sigma.
    """
    observed = audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds)
    obs_sigma = observed["sigma"]

    rng = np.random.default_rng(42)
    mask = (pulsar_times >= gw_time - window_seconds) & (pulsar_times <= gw_time + window_seconds)
    n_in_window = int(np.sum(mask))

    if n_in_window == 0 or len(pulsar_residuals) < 2:
        return {"observed_sigma": obs_sigma, "p_value": 1.0,
                "n_iterations": n_iterations, "n_exceed": 0}

    baseline_idx = ~mask
    n_exceed = 0
    for _ in range(n_iterations):
        shuffled = rng.permutation(pulsar_residuals)
        w_rms = np.sqrt(np.mean(shuffled[mask] ** 2))
        b_res = shuffled[baseline_idx]
        b_rms = np.sqrt(np.mean(b_res ** 2))
        b_std = np.std(b_res)
        trial_sigma = (w_rms - b_rms) / b_std if b_std > 0 else 0
        if trial_sigma >= obs_sigma:
            n_exceed += 1

    return {"observed_sigma": obs_sigma, "p_value": n_exceed / n_iterations,
            "n_iterations": n_iterations, "n_exceed": n_exceed}


# ---------------------------------------------------------------------------
# Sliding Window Sweep
# ---------------------------------------------------------------------------

def sliding_window_sweep(gw_time, pulsar_times, pulsar_residuals,
                         window_seconds=10.0, n_offsets=200):
    """
    Sweep an analysis window across the timeline and compute sigma at each offset.
    """
    t_min, t_max = float(np.min(pulsar_times)), float(np.max(pulsar_times))
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

    gw_result = audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds)
    return {"offsets": offsets, "sigmas": sigmas, "gw_sigma": gw_result["sigma"]}


# ---------------------------------------------------------------------------
# Null Distribution Test
# ---------------------------------------------------------------------------

def null_distribution_test(pulsar_times, pulsar_residuals, window_seconds=10.0,
                           n_trials=100, seed=123):
    """
    Run the audit at n_trials random GPS times (no GW event) to build
    a null distribution of sigma values.

    Returns dict with:
      null_sigmas  — array of sigma values from random window placements
      mean         — mean of the null distribution
      std          — std of the null distribution
    """
    rng = np.random.default_rng(seed)
    t_min = float(np.min(pulsar_times)) + window_seconds
    t_max = float(np.max(pulsar_times)) - window_seconds

    if t_max <= t_min:
        return {"null_sigmas": np.zeros(n_trials), "mean": 0.0, "std": 0.0}

    random_times = rng.uniform(t_min, t_max, n_trials)
    null_sigmas = np.empty(n_trials)

    for i, t in enumerate(random_times):
        result = audit_lag(t, pulsar_times, pulsar_residuals, window_seconds)
        null_sigmas[i] = result["sigma"]

    return {
        "null_sigmas": null_sigmas,
        "mean": float(np.mean(null_sigmas)),
        "std": float(np.std(null_sigmas)),
    }


# ---------------------------------------------------------------------------
# Correlation Matrix (pairwise pulsar residual correlations)
# ---------------------------------------------------------------------------

def correlation_matrix(ensemble_data, gw_time, window_seconds=10.0):
    """
    Compute pairwise Pearson correlation between pulsar residuals
    within the analysis window. A simulation artifact should produce
    correlated glitches across unrelated pulsars.

    Returns dict with:
      names       — list of pulsar names
      matrix      — 2D numpy array of correlation coefficients
    """
    names = list(ensemble_data.keys())
    n = len(names)
    matrix = np.eye(n)

    # Build per-pulsar windowed residuals interpolated onto a common grid
    windowed = {}
    for psr, (times, residuals) in ensemble_data.items():
        mask = (times >= gw_time - window_seconds) & (times <= gw_time + window_seconds)
        if np.any(mask):
            windowed[psr] = (times[mask], residuals[mask])
        else:
            windowed[psr] = (np.array([]), np.array([]))

    # Common time grid (union range, 500 points)
    all_t_min = min((t.min() for t, _ in windowed.values() if len(t) > 0), default=gw_time - window_seconds)
    all_t_max = max((t.max() for t, _ in windowed.values() if len(t) > 0), default=gw_time + window_seconds)
    common_t = np.linspace(all_t_min, all_t_max, 500)

    # Interpolate each pulsar onto common grid
    interp_residuals = {}
    for psr in names:
        t, r = windowed[psr]
        if len(t) >= 2:
            interp_residuals[psr] = np.interp(common_t, t, r)
        else:
            interp_residuals[psr] = np.zeros(500)

    for i in range(n):
        for j in range(i + 1, n):
            r_i = interp_residuals[names[i]]
            r_j = interp_residuals[names[j]]
            if np.std(r_i) > 0 and np.std(r_j) > 0:
                corr = np.corrcoef(r_i, r_j)[0, 1]
            else:
                corr = 0.0
            matrix[i, j] = corr
            matrix[j, i] = corr

    return {"names": names, "matrix": matrix}


# ---------------------------------------------------------------------------
# Hellings-Downs Curve
# ---------------------------------------------------------------------------

def hellings_downs_analysis(ensemble_data, pulsar_coords, gw_time, window_seconds=10.0):
    """
    Compute cross-correlations vs angular separations between pulsar pairs
    and compare against the theoretical Hellings-Downs curve.

    Args:
        ensemble_data: dict {psr_name: (times, residuals)}
        pulsar_coords: dict {psr_name: (ra_deg, dec_deg)}
        gw_time: GPS time of GW event
        window_seconds: analysis window

    Returns dict with:
      angles_deg    — array of angular separations (degrees)
      correlations  — array of measured cross-correlations
      hd_curve_x    — theoretical HD curve x-values (degrees)
      hd_curve_y    — theoretical HD curve y-values
    """
    names = [p for p in ensemble_data if p in pulsar_coords]
    n = len(names)

    angles = []
    correlations = []

    # Build interpolated windowed residuals
    windowed = {}
    for psr in names:
        times, residuals = ensemble_data[psr]
        mask = (times >= gw_time - window_seconds) & (times <= gw_time + window_seconds)
        if np.any(mask):
            windowed[psr] = (times[mask], residuals[mask])
        else:
            windowed[psr] = (np.array([]), np.array([]))

    all_t = [t for t, _ in windowed.values() if len(t) > 0]
    if not all_t:
        return {"angles_deg": np.array([]), "correlations": np.array([]),
                "hd_curve_x": np.array([]), "hd_curve_y": np.array([])}

    t_min = min(t.min() for t in all_t)
    t_max = max(t.max() for t in all_t)
    common_t = np.linspace(t_min, t_max, 500)

    interp = {}
    for psr in names:
        t, r = windowed[psr]
        interp[psr] = np.interp(common_t, t, r) if len(t) >= 2 else np.zeros(500)

    for i in range(n):
        for j in range(i + 1, n):
            # Angular separation
            ra1, dec1 = np.radians(pulsar_coords[names[i]])
            ra2, dec2 = np.radians(pulsar_coords[names[j]])
            cos_sep = (np.sin(dec1) * np.sin(dec2) +
                       np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
            cos_sep = np.clip(cos_sep, -1, 1)
            sep_deg = np.degrees(np.arccos(cos_sep))
            angles.append(sep_deg)

            # Cross-correlation
            r_i = interp[names[i]]
            r_j = interp[names[j]]
            if np.std(r_i) > 0 and np.std(r_j) > 0:
                corr = np.corrcoef(r_i, r_j)[0, 1]
            else:
                corr = 0.0
            correlations.append(corr)

    # Theoretical Hellings-Downs curve
    hd_x = np.linspace(0.01, 180, 200)
    hd_y = _hellings_downs_value(np.radians(hd_x))

    return {
        "angles_deg": np.array(angles),
        "correlations": np.array(correlations),
        "hd_curve_x": hd_x,
        "hd_curve_y": hd_y,
    }


def _hellings_downs_value(theta):
    """
    Hellings-Downs overlap reduction function.
    Γ(θ) = (1/2) - (1/4) * x + (3/2) * x * ln(x)
    where x = (1 - cos θ) / 2
    """
    x = (1 - np.cos(theta)) / 2
    # Avoid log(0) for θ=0
    result = np.where(x > 0, 0.5 - 0.25 * x + 1.5 * x * np.log(x), 0.5)
    return result


# ---------------------------------------------------------------------------
# Bayesian Evidence Ratio
# ---------------------------------------------------------------------------

def bayesian_evidence_ratio(gw_time, pulsar_times, pulsar_residuals,
                            window_seconds=10.0, n_samples=10000):
    """
    Compute a Bayes factor comparing "artifact present" vs "null" models.

    Uses a simple approach:
      - Null model: residuals drawn from N(0, baseline_std)
      - Artifact model: residuals drawn from N(0, elevated_std) in window

    The Bayes factor K = P(data|artifact) / P(data|null).
    K > 1 favors artifact, K < 1 favors null.

    Returns dict with:
      bayes_factor    — K value
      log_bf          — log10(K)
      interpretation  — human-readable string
    """
    mask = (pulsar_times >= gw_time - window_seconds) & (pulsar_times <= gw_time + window_seconds)
    w_res = pulsar_residuals[mask]
    b_res = pulsar_residuals[~mask]

    if len(w_res) == 0 or len(b_res) < 2:
        return {"bayes_factor": 1.0, "log_bf": 0.0, "interpretation": "Insufficient data"}

    baseline_std = np.std(b_res)
    window_std = np.std(w_res)

    if baseline_std <= 0:
        return {"bayes_factor": 1.0, "log_bf": 0.0, "interpretation": "No variance in baseline"}

    n_w = len(w_res)

    # Log-likelihood under null model: all data from baseline distribution
    ll_null = -0.5 * n_w * np.log(2 * np.pi * baseline_std**2) - \
              np.sum(w_res**2) / (2 * baseline_std**2)

    # Log-likelihood under artifact model: window data from elevated distribution
    ll_artifact = -0.5 * n_w * np.log(2 * np.pi * window_std**2) - \
                  np.sum(w_res**2) / (2 * window_std**2)

    # Log Bayes factor
    log_bf = (ll_artifact - ll_null) / np.log(10)  # Convert to log10

    # Clamp for numerical stability
    log_bf = float(np.clip(log_bf, -100, 100))
    bf = 10 ** log_bf

    # Interpretation (Jeffreys' scale)
    if log_bf < -2:
        interp = "Decisive evidence for NULL"
    elif log_bf < -1:
        interp = "Strong evidence for NULL"
    elif log_bf < -0.5:
        interp = "Moderate evidence for NULL"
    elif log_bf < 0.5:
        interp = "Inconclusive"
    elif log_bf < 1:
        interp = "Moderate evidence for ARTIFACT"
    elif log_bf < 2:
        interp = "Strong evidence for ARTIFACT"
    else:
        interp = "Decisive evidence for ARTIFACT"

    return {"bayes_factor": bf, "log_bf": log_bf, "interpretation": interp}


# ---------------------------------------------------------------------------
# Ensemble Audit (with Monte Carlo, Bayesian, null distribution)
# ---------------------------------------------------------------------------

def audit_ensemble(gw_time, ensemble_data, window_seconds=10.0,
                   monte_carlo=True, mc_iterations=10000):
    """
    Performs a coherent audit across an ensemble of pulsars.
    Includes Monte Carlo p-values and Bayesian evidence per pulsar.
    """
    results = {}
    all_sigmas = []

    for psr_name, (times, residuals) in ensemble_data.items():
        res = audit_lag(gw_time, times, residuals, window_seconds)

        # Monte Carlo p-value
        if monte_carlo:
            mc = monte_carlo_test(gw_time, times, residuals,
                                  window_seconds, n_iterations=mc_iterations)
            res["p_value"] = mc["p_value"]
        else:
            res["p_value"] = None

        # Bayesian evidence
        bf = bayesian_evidence_ratio(gw_time, times, residuals, window_seconds)
        res["bayes_factor"] = bf["bayes_factor"]
        res["log_bf"] = bf["log_bf"]
        res["bf_interpretation"] = bf["interpretation"]

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
