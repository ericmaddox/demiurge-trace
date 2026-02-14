import matplotlib.pyplot as plt
import numpy as np

def plot_results(gw_time, pulsar_times, pulsar_residuals, analysis_results, gw_strain=None, output_file=None):
    """
    Plots the pulsar residuals and highlights the GW event window.
    Optionally plots GW strain data if provided.
    """
    
    if gw_strain is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Plot GW Strain
        strain_times = gw_strain.times.value - gw_time
        ax1.plot(strain_times, gw_strain.value, color='purple', label='GW Strain (H1)', alpha=0.8)
        ax1.set_ylabel('Strain')
        ax1.set_title(f"Gravitational Wave Event: {analysis_results['gw_time']}")
        ax1.legend(loc='upper right')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add event line
        ax1.axvline(0, color='red', linestyle='-', linewidth=2)
        
        # Plot Residuals on ax2
        ax = ax2
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot residuals
    ax.plot(pulsar_times - gw_time, pulsar_residuals * 1e6, '.', label='Pulsar Residuals', alpha=0.5)
    
    # Horizontal line for 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # Vertical line for GW Event
    ax.axvline(0, color='red', linestyle='-', linewidth=2, label='GW Event Merger')
    
    # Highlight lag spikes if detected
    if analysis_results['is_artifact']:
        ax.axvspan(-10, 10, color='yellow', alpha=0.3, label='Potential Simulation Artifact')
        plt.suptitle(f"SIMULATION ARTIFACT DETECTED! (Sigma: {analysis_results['sigma']:.2f})", color='red', weight='bold')
    else:
        plt.suptitle(f"Null Hypothesis Validated (Sigma: {analysis_results['sigma']:.2f})", color='green')
        
    ax.set_xlabel('Time relative to GW Event (s)')
    ax.set_ylabel('Residuals (microseconds)')
    ax.legend(loc='upper right')
    # grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Zoom in to interesting area
    ax.set_xlim(-16, 16) # Zoom to +/- 16s to see the wave clearly if duration is 32s
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_ensemble_results(gw_time, ensemble_data, ensemble_results, gw_strain=None, output_file=None):
    """
    Plots normalized residuals for the entire ensemble.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # 1. Plot GW Strain
    if gw_strain is not None:
        strain_times = gw_strain.times.value - gw_time
        ax1.plot(strain_times, gw_strain.value, color='purple', label='GW Strain (H1)', alpha=0.8)
        ax1.set_ylabel('Strain')
        ax1.set_title(f"Universal Clock Ensemble Audit: {ensemble_results['gw_time']}")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axvline(0, color='red', linewidth=2)
    
    # 2. Plot Ensemble Residuals
    for psr_name, (times, residuals) in ensemble_data.items():
        # Z-score normalization for plotting together
        # (Residual - Mean) / Std
        std = np.std(residuals)
        if std > 0:
            norm_res = residuals / std
        else:
            norm_res = residuals
            
        ax2.plot(times - gw_time, norm_res, '.', label=psr_name, alpha=0.4, markersize=3)
        
    ax2.axhline(0, color='black', alpha=0.8)
    ax2.axvline(0, color='red', linewidth=2, label='GW Merger')
    
    if ensemble_results['is_ensemble_artifact']:
        ax2.axvspan(-10, 10, color='yellow', alpha=0.3)
        plt.suptitle(f"COHERENT ENSEMBLE ARTIFACT DETECTED! (Score: {ensemble_results['ensemble_sigma']:.2f})", color='red', fontsize=16, weight='bold')
    else:
        plt.suptitle(f"Universal Clock Stability Confirmed (Ensemble Score: {ensemble_results['ensemble_sigma']:.2f})", color='green', fontsize=16)

    ax2.set_xlabel('Time relative to GW Event (s)')
    ax2.set_ylabel('Normalized Deviation (sigma)')
    ax2.set_xlim(-16, 16)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Shrink legend if too many
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

