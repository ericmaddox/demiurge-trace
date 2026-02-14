<p align="center">
  <h1 align="center">🔭 Demiurge Trace</h1>
  <p align="center">
    <strong>A high-precision astrophysical auditing tool for testing the Simulation Hypothesis.</strong>
  </p>
  <p align="center">
    Correlates LIGO/Virgo/KAGRA gravitational wave events with sub-microsecond pulsar timing residuals<br/>
    from the NANOGrav 15-year dataset to search for coherent "simulation lag" artifacts.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="https://gwosc.org/"><img src="https://img.shields.io/badge/Data-LIGO_GWOSC-E67E22?style=flat-square" alt="LIGO GWOSC"></a>
  <a href="https://nanograv.org/"><img src="https://img.shields.io/badge/Data-NANOGrav_15yr-8E44AD?style=flat-square" alt="NANOGrav 15yr"></a>
  <a href="https://github.com/nanograv/PINT"><img src="https://img.shields.io/badge/Timing-PINT-2ECC71?style=flat-square" alt="PINT"></a>
  <a href="https://gwpy.github.io/"><img src="https://img.shields.io/badge/GW-GWpy-3498DB?style=flat-square" alt="GWpy"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [The Hypothesis](#the-hypothesis)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Usage](#usage)
  - [Ensemble Audit](#ensemble-audit)
  - [Single Pulsar Audit](#single-pulsar-audit)
  - [Simulation Mode](#simulation-mode)
  - [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Scientific Details](#scientific-details)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Overview

**Demiurge Trace** bridges two domains of astrophysics:

- **High-energy physics** — Confirmed gravitational wave merger events detected by LIGO/Virgo/KAGRA.
- **Precision timing** — Sub-microsecond timing residuals from millisecond pulsars (MSPs) in the NANOGrav 15-year dataset.

The tool performs a statistical audit to determine whether coherent timing anomalies appear across a pulsar array during gravitational wave events — a signature that would be consistent with computational "lag" in a simulated universe.

---

## The Hypothesis

If our universe operates on finite computational resources, extremely high-load events — such as binary neutron star or black hole mergers — could theoretically produce measurable "simulation lag." This lag would manifest as a coherent, synchronized timing deviation across the galaxy's most stable natural clocks: millisecond pulsars.

A genuine simulation artifact would be:

1. **Temporally Correlated** — Occurring precisely within the GW merger event window.
2. **Spatially Invariant** — Affecting the entire observable pulsar array simultaneously, regardless of sky position.

Unlike individual pulsar jitter or instrumentation noise, a coherent artifact would stand out as a statistically significant ensemble-wide deviation.

---

## Architecture

The system is built as a modular pipeline with four stages:

```mermaid
graph TD
    subgraph "Data Acquisition"
        A1[LIGO GWOSC Catalog] -->|gwpy| B1(GW Strain Data)
        A2[NANOGrav 15-Year Dataset] -->|setup_array.py| B2(Pulsar .par/.tim Files)
    end

    subgraph "Processing Engine"
        B2 --> C1{Pulsar Module}
        C1 -->|PINT| D1[High-Precision Residuals]
        B1 --> C2[LIGO Module]
        C2 -->|Signal Analysis| D2[Event GPS Timestamps]
    end

    subgraph "Analysis & Verification"
        D1 & D2 --> E[Auditor Engine]
        E -->|audit_ensemble| F{Statistical Audit}
        F -->|> 3σ| G[🚨 Coherent Artifact]
        F -->|< 3σ| H[✓ Null Hypothesis]
    end

    subgraph "Visualization"
        H & G --> I[Visualizer]
        I -->|Matplotlib| J[Result Plots]
    end
```

---

## Getting Started

### Prerequisites

| Dependency | Purpose |
|:---|:---|
| [Python 3.9+](https://www.python.org/downloads/) | Runtime |
| [PINT](https://github.com/nanograv/PINT) | Pulsar timing model loading & residual calculation |
| [GWpy](https://gwpy.github.io/) | LIGO/GWOSC data access & strain retrieval |
| [Matplotlib](https://matplotlib.org/) | Visualization |
| [Astropy](https://www.astropy.org/) | Time coordinate transformations (GPS ↔ MJD/TDB) |

### Installation

```bash
# Clone the repository
git clone https://github.com/ericmaddox/demiurge-trace.git
cd demiurge-trace

# Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

# Install dependencies
pip install pint-pulsar gwpy matplotlib numpy requests dash plotly
```

### Data Setup

The auditor requires real millisecond pulsar timing data. Run the setup script to automatically download, extract, and patch the NANOGrav 15-year narrowband dataset from Zenodo:

```bash
python setup_array.py
```

This script handles:
- Downloading the dataset tarball (~600 MB) from [Zenodo Record 8423265](https://zenodo.org/records/8423265)
- Extracting `.par` and `.tim` files for the target pulsar ensemble
- Patching binary model parameters for PINT compatibility (e.g., `T2` → `DDK`)

Data is saved to `data/real/`.

---

## Usage

### Ensemble Audit

Run the full pulsar array audit against a specific gravitational wave event:

```bash
python main.py --event GW170817 --ensemble --window 432000 --output results/audit_GW170817.png
```

> **Note:** A window of `432000` seconds (5 days) is recommended to account for the sparse cadence of pulsar timing array observations.

### Single Pulsar Audit

Audit a single pulsar against an event:

```bash
python main.py --event GW150914 --pulsar J1713+0747
```

### Simulation Mode

Inject a synthetic 10 µs lag spike across the ensemble to validate detection sensitivity:

```bash
python main.py --event GW170817 --ensemble --simulate-lag --output results/sim_detection.png
```

### Interactive Dashboard

Launch the web-based Plotly Dash dashboard for interactive exploration:

```bash
python dashboard.py
```

Your default browser will open automatically once data pre-loading completes (~60s).

### Reading the Dashboard

After clicking **▶ Run Audit**, the dashboard populates three tabs:

#### Tab 1 — Ensemble Overview

| Element | What it shows |
|:---|:---|
| **Top subplot (purple)** | Raw LIGO strain data from the H1 (Hanford) detector centered on the GW event. The red vertical line marks the exact merger time (t = 0). |
| **Bottom subplot (colored dots)** | Timing residuals for all pulsars in the ensemble, normalized to units of σ (standard deviations). Each color is a different pulsar. |
| **Yellow shaded region** | The analysis window — the time range being compared to the baseline. |
| **What to look for** | If the simulation hypothesis were true, you'd see a *coordinated cluster* of elevated residuals (dots far from y = 0) inside the yellow window across multiple pulsars simultaneously. Scattered, random dots are normal — that's pulsar noise. |

#### Tab 2 — Per-Pulsar Detail

| Element | What it shows |
|:---|:---|
| **Grey dots** | Baseline residuals (outside the analysis window) in microseconds (µs). This is the pulsar's normal timing noise. |
| **Purple dots** | Residuals *inside* the analysis window, highlighted for comparison against the baseline. |
| **Title bar stats** | `σ` = how many standard deviations the window RMS deviates from the baseline RMS. `Window RMS` = root-mean-square of residuals inside the window. `Baseline RMS` = the same metric for all data outside the window. |
| **What to look for** | If σ ≈ 0, the window residuals are indistinguishable from normal noise — no artifact. If σ > 3, the window shows statistically significant excess timing variation that warrants investigation. |

#### Tab 3 — Sliding Window Sweep

| Element | What it shows |
|:---|:---|
| **Colored lines** | Each line is one pulsar's σ deviation as the analysis window slides across the entire observation timeline. The x-axis is time offset from the GW event in days. |
| **Red vertical line** | Marks the exact GW event time (offset = 0). |
| **Yellow dashed line** | The 3σ threshold. Any spike above this line would be statistically significant. |
| **What to look for** | If no pulsar's line spikes at offset = 0, the GW event time is not special — the residuals there are no different from any other random time in the dataset. A coordinated spike at offset = 0 across multiple pulsars would suggest a simulation artifact. |

#### Tab 4 — Results Table

| Column | Meaning |
|:---|:---|
| **Pulsar** | NANOGrav pulsar designation (J-name). |
| **σ Deviation** | Number of standard deviations between window RMS and baseline RMS. Values near 0 = normal; > 3 = anomalous. |
| **p-value** | Monte Carlo significance (10,000 permutation trials). The fraction of random residual shuffles that produce a sigma ≥ the observed value. Low p-values (< 0.05) suggest the deviation is unlikely due to chance. |
| **Window RMS** | Root-mean-square of timing residuals inside the analysis window (seconds). |
| **Baseline RMS** | Root-mean-square of residuals outside the window (seconds). |
| **Artifact** | ✓ No = consistent with real physics. 🚨 YES = anomalous deviation detected. |
| **Verdict banner** | Ensemble-level conclusion. "Universal Clock Stability Confirmed" means no coherent artifact was found across the pulsar array. |

> **Interpreting results:** A σ near 0 across all pulsars confirms that the universe's natural clocks (pulsars) show no timing anomalies coincident with the gravitational wave event — consistent with real physics, not a simulation.

### CLI Reference

```
usage: main.py [-h] [--event EVENT] [--pulsar PULSAR] [--ensemble]
               [--window WINDOW] [--simulate-lag] [--output OUTPUT]

Demiurge Trace — Simulation Hypothesis Auditor

options:
  --event EVENT         GW event name (default: GW150914)
  --pulsar PULSAR       Pulsar name for single-pulsar mode (default: J1713+0747)
  --ensemble            Run audit on the entire pulsar array
  --window WINDOW       Analysis window in seconds (default: 86400)
  --simulate-lag        Inject a synthetic lag spike for testing
  --output OUTPUT       Output plot filename (default: audit_result.png)
```

---

## Project Structure

```
demiurge-trace/
├── main.py                  # CLI entry point
├── dashboard.py             # Interactive Plotly Dash web dashboard
├── setup_array.py           # NANOGrav data acquisition & patching
├── setup_data.py            # Alternative data setup utilities
├── get_data_urls.py         # Zenodo URL resolver
├── dashboard_figures.py     # Plotly figure builders (sky map, heatmap, etc.)
├── demiurge_trace/          # Core library
│   ├── __init__.py
│   ├── ligo_module.py       # GW event lookup & strain data fetching
│   ├── pulsar_module.py     # PINT model loading & residual calculation
│   ├── auditor.py           # Statistical audit engine (MC, Bayes, HD, etc.)
│   └── visualizer.py        # Matplotlib plotting (single + ensemble)
├── .github/workflows/ci.yml # GitHub Actions CI smoke test
├── data/
│   └── real/                # NANOGrav .par/.tim files (generated by setup)
├── results/                 # Output plots
└── tools/                   # Diagnostic & debugging utilities
```

---

## Scientific Details

### Timing Precision

- **Planetary Shapiro Correction:** TOAs are loaded with `planets=True` to include Solar System ephemeris data (DE421), essential for sub-microsecond residual accuracy.
- **BIPM Clock Corrections:** Observatory clock corrections are applied using the latest BIPM2023 timescale.

### Time Coordinate Alignment

All LIGO timestamps (GPS) are synchronized with pulsar timing coordinates (MJD/TDB) using `astropy.time`, maintaining sub-millisecond alignment across the two data domains.

### Model Compatibility

The data setup pipeline includes an automated patching layer to handle non-standard NANOGrav parameter names and binary models (e.g., `T2` → `DDK` conversion) for full PINT compatibility.

### Statistical Method

The auditor computes timing residual RMS within a configurable window around each GW event and compares it against the baseline RMS distribution. An ensemble-wide sigma score is derived by averaging individual pulsar deviations. A coherent artifact is flagged when:

- The ensemble sigma exceeds **3.0σ**, or
- Three or more pulsars simultaneously exceed **1.0σ** (in arrays of ≥ 5 pulsars)

### Monte Carlo Significance Testing

Each pulsar's observed sigma is validated against **10,000 random permutation trials**. Residuals are shuffled (breaking the time association), and the window sigma is recomputed for each trial. The **p-value** is the fraction of trials producing a sigma ≥ the observed value — providing a rigorous, distribution-free measure of statistical significance.

### Sliding Window Sweep

The analysis window is swept across the entire observation timeline at 200 evenly-spaced positions. This produces a sigma-vs-offset curve for each pulsar. If the GW event time is unremarkable, the sigma at offset = 0 should be indistinguishable from the background — confirming that the event coincidence is not special.

### Pulsar Ensemble

| Pulsar | Type | Notes |
|:---|:---|:---|
| J1713+0747 | MSP | DDK binary model |
| J1909-3744 | MSP | High-precision timing standard |
| J0437-4715 | MSP | Closest and brightest MSP |
| J1614-2230 | MSP | Massive neutron star |
| J1744-1134 | MSP | Isolated MSP |
| B1937+21 | MSP | Original millisecond pulsar |
| B1855+09 | MSP | Long timing baseline |
| J1600-3053 | MSP | Wideband timing target |
| J2145-0750 | MSP | Southern sky pulsar |
| J1857+0943 | MSP | NANOGrav array member |

---

## Contributing

Contributions are welcome! Here are some areas where help is needed:

- **Multi-event sweep** — Automate auditing across the full GWOSC catalog (~90 events)
- **Null distribution test** — Audit at random GPS times to build a baseline sigma distribution
- **Cross-correlation analysis** — Pairwise pulsar residual correlations (Hellings-Downs style)
- **Sky map visualization** — Mollweide projection color-coded by sigma
- **Multi-detector strain** — Add L1 (Livingston) and V1 (Virgo) alongside H1
- **Additional PTA data** — EPTA, PPTA, and IPTA integration

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is currently unlicensed. See [choosealicense.com](https://choosealicense.com/) to select an appropriate open-source license.

---

## Disclaimer

This project is an experimental tool for testing the limits of astrophysical data in the context of the Simulation Hypothesis. It is **not** intended as a proof of simulation, but as a rigorous statistical audit of physical consistency using real observational data.

---

## Changelog

### v0.4.1 — 2026-02-14

- **Fixed Sliding Window tab** — resolved duplicate `hovermode` keyword causing Plotly rendering failure
- **Fixed Sky Map tab** — replaced `Scattergeo`/Mollweide projection with RA/DEC scatter for Plotly compatibility
- **Fixed Correlations tab** — converted numpy matrix to plain lists for proper heatmap text rendering
- **All 8 dashboard tabs** now render correctly after audit

### v0.4.0 — 2026-02-14

- **Sky Map tab** — Mollweide projection of all pulsars color-coded by σ deviation
- **Correlation Matrix tab** — Pairwise Pearson correlation heatmap between pulsar residuals
- **Null Distribution tab** — Histogram of ensemble σ at 100 random GPS times vs observed
- **Hellings-Downs tab** — Angular separation vs cross-correlation with theoretical HD curve
- **Bayesian Evidence Ratio** — Bayes factor (log₁₀ BF) and Jeffreys' scale verdict per pulsar
- **CSV/JSON export** — Download results directly from the dashboard sidebar
- **GitHub Actions CI** — Automated smoke tests on push/PR with synthetic data
- **Code refactor** — Figure builders extracted to `dashboard_figures.py`
- **8-tab dashboard** — Ensemble, Per-Pulsar, Sliding Window, Sky Map, Correlations, Null Dist, Hellings-Downs, Results

### v0.3.0 — 2026-02-14

- **Monte Carlo significance testing** — 10,000-iteration permutation test per pulsar with p-values
- **Sliding window sweep** — New dashboard tab showing σ vs time-offset across the full timeline
- **p-value column** added to Results Table for statistical rigor

### v0.2.0 — 2026-02-14

- **Interactive Plotly Dash dashboard** with 4 tabs (Ensemble, Per-Pulsar, Sliding Window, Results)
- **Fixed missing pulsar J1857+0943** — alias mapping for NANOGrav B-name `B1857+09`
- **Server-side data architecture** — eliminated JSON serialization bottleneck
- **Auto-open browser** on dashboard startup

### v0.1.0 — 2026-02-08

- Initial release with CLI-based auditor
- NANOGrav 15-year data acquisition (`setup_array.py`)
- LIGO GWOSC strain data integration
- 9-pulsar ensemble analysis for GW150914 and GW170817
- Professional README with badges and documentation
