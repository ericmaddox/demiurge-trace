import argparse
import sys
from demiurge_trace import ligo_module, pulsar_module, auditor, visualizer

def main():
    parser = argparse.ArgumentParser(description="Demiurge Trace")
    parser.add_argument("--event", type=str, default="GW150914", help="GW Event Name (e.g., GW150914)")
    parser.add_argument("--pulsar", type=str, default="J1713+0747", help="Pulsar name for single-pulsar mode (e.g., J1713+0747)")
    parser.add_argument("--ensemble", action="store_true", help="Run audit on the entire pulsar array")
    parser.add_argument("--window", type=float, default=86400.0, help="Analysis window in seconds (default: 86400s / 1 day)")
    parser.add_argument("--simulate-lag", action="store_true", help="Inject a fake lag spike for testing")

    parser.add_argument("--output", type=str, default="audit_result.png", help="Output plot filename")
    
    args = parser.parse_args()
    
    print(f"Initializing Demiurge Trace for Event: {args.event}")
    
    # 1. Fetch GW Event Time
    print("Fetching GW Event Timestamp...")
    gw_time = ligo_module.get_gw_event_time(args.event)
    if not gw_time:
        print("Failed to fetch GW event time.")
        sys.exit(1)
    print(f"Event Time (GPS): {gw_time}")
    
    # 1.5 Fetch GW Strain Data
    print("Fetching GW Strain Data for visualization...")
    gw_strain = ligo_module.fetch_strain_data(gw_time)
    
    if args.ensemble:
        # Ensemble Mode
        print("ENTERING ENSEMBLE AUDIT MODE (Full Pulsar Array)...")
        ensemble_data = pulsar_module.get_ensemble_data(gw_time, is_simulation=args.simulate_lag)
        
        if not ensemble_data:
            print("No pulsar data available for ensemble.")
            sys.exit(1)
            
        print(f"Successfully loaded {len(ensemble_data)} pulsars.")
        results = auditor.audit_ensemble(gw_time, ensemble_data, window_seconds=args.window)
        
        print("-" * 30)
        print(f"ENSEMBLE ANALYSIS RESULTS:")
        print(f"Pulsars in Window: {results['n_pulsars_in_window']}")
        print(f"Mean Ensemble Sigma: {results['ensemble_sigma']:.2f}")
        print(f"Pulsars with > 1σ deviation: {results['n_detectors_above_1s']}")
        print(f"COHERENT ARTIFACT DETECTED: {results['is_ensemble_artifact']}")
        print("-" * 30)
        
        visualizer.plot_ensemble_results(gw_time, ensemble_data, results, gw_strain=gw_strain, output_file=args.output)
    else:
        # Single Pulsar Mode
        print(f"Loading Pulsar Data for {args.pulsar}...")
        pulsar_times, pulsar_residuals = pulsar_module.get_pulsar_data(args.pulsar, gw_time, is_simulation=args.simulate_lag)
        
        # 3. Audit Lag
        print("Auditing for Simulation Lag...")
        results = auditor.audit_lag(gw_time, pulsar_times, pulsar_residuals, window_seconds=args.window)

        
        print("-" * 30)
        print(f"Analysis Results ({args.pulsar}):")
        print(f"Window RMS: {results['window_rms']:.2e} s")
        print(f"Baseline RMS: {results['baseline_rms']:.2e} s")
        print(f"Sigma Deviation: {results['sigma']:.2f}")
        print(f"Artifact Detected: {results['is_artifact']}")
        print("-" * 30)
        
        # 4. Visualize
        print(f"Generating Visualization: {args.output}")
        visualizer.plot_results(gw_time, pulsar_times, pulsar_residuals, results, gw_strain=gw_strain, output_file=args.output)

    print("Done.")

if __name__ == "__main__":
    main()
