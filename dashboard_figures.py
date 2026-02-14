"""
Dashboard figure builders — separated for maintainability.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "bg": "#0a0e17", "card": "#111827", "card_border": "#1e293b",
    "accent": "#6366f1", "accent_glow": "rgba(99,102,241,0.15)",
    "text": "#e2e8f0", "text_dim": "#94a3b8", "green": "#10b981",
    "red": "#ef4444", "yellow": "#f59e0b", "purple": "#a78bfa", "strain": "#c084fc",
}
PULSAR_COLORS = [
    "#6366f1", "#ec4899", "#14b8a6", "#f59e0b", "#ef4444",
    "#8b5cf6", "#06b6d4", "#84cc16", "#f97316", "#64748b",
]

# Known pulsar sky coordinates (J2000 RA/DEC in degrees)
PULSAR_COORDS = {
    "J1713+0747": (258.25, 7.79), "J1909-3744": (287.25, -37.74),
    "J0437-4715": (69.32, -47.25), "J1614-2230": (243.65, -22.50),
    "J1744-1134": (266.12, -11.57), "B1937+21": (294.91, 21.58),
    "B1855+09": (284.21, 9.72), "J1600-3053": (240.08, -30.88),
    "J2145-0750": (326.46, -7.84), "J1857+0943": (284.42, 9.73),
}


def _base_layout(height=500):
    return dict(
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
        height=height, margin=dict(l=60, r=30, t=60, b=40), hovermode="closest",
    )


def empty_fig(msg="Press ▶ Run Audit to begin."):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color=COLORS["text_dim"]))
    fig.update_layout(**_base_layout(), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def build_ensemble_fig(ensemble, strain, gw_time, window):
    if not ensemble:
        return empty_fig("No data loaded.")
    has_strain = strain is not None
    rows = 2 if has_strain else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.3, 0.7] if has_strain else [1],
                        subplot_titles=(["GW Strain (H1)", "Pulsar Timing Residuals (σ)"]
                                        if has_strain else ["Pulsar Timing Residuals (σ)"]))
    if has_strain:
        st = strain.times.value - gw_time
        fig.add_trace(go.Scattergl(x=st, y=strain.value, mode="lines",
                                   line=dict(color=COLORS["strain"], width=1), name="Strain"), row=1, col=1)
        fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2), row=1, col=1)
    cr = 2 if has_strain else 1
    for i, (psr, (times, res)) in enumerate(ensemble.items()):
        t = times - gw_time
        std = np.std(res)
        norm = (res / std) if std > 0 else res
        fig.add_trace(go.Scattergl(x=t, y=norm, mode="markers",
            marker=dict(size=3, color=PULSAR_COLORS[i % len(PULSAR_COLORS)], opacity=0.5),
            name=psr, hovertemplate=f"<b>{psr}</b><br>Δt=%{{x:.1f}}s<br>σ=%{{y:.2f}}<extra></extra>"),
            row=cr, col=1)
    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2), row=cr, col=1)
    fig.add_vrect(x0=-window, x1=window, fillcolor="rgba(245,158,11,0.06)",
                  line=dict(width=1, color="rgba(245,158,11,0.3)"), row=cr, col=1)
    fig.update_layout(**_base_layout(700),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11)))
    fig.update_xaxes(title_text="Time relative to GW Event (s)", row=cr, col=1, gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def build_pulsar_fig(ensemble, gw_time, psr_name, window, audit_results):
    if psr_name not in ensemble:
        return empty_fig("Pulsar not found.")
    times, residuals = ensemble[psr_name]
    t = times - gw_time
    res_us = residuals * 1e6
    mask = (t >= -window) & (t <= window)
    psr_res = audit_results.get("pulsar_results", {}).get(psr_name, {})
    sigma = psr_res.get("sigma", 0)
    w_rms = psr_res.get("window_rms", 0)
    b_rms = psr_res.get("baseline_rms", 0)
    p_val = psr_res.get("p_value")
    p_str = f"  |  p = {p_val:.4f}" if p_val is not None else ""
    log_bf = psr_res.get("log_bf", 0)
    fig = go.Figure()
    if np.any(~mask):
        fig.add_trace(go.Scattergl(x=t[~mask], y=res_us[~mask], mode="markers",
            marker=dict(size=4, color=COLORS["text_dim"], opacity=0.3), name="Baseline"))
    if np.any(mask):
        fig.add_trace(go.Scattergl(x=t[mask], y=res_us[mask], mode="markers",
            marker=dict(size=6, color=COLORS["accent"], opacity=0.8,
                        line=dict(width=1, color="white")), name="In Window"))
    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2))
    fig.add_vrect(x0=-window, x1=window, fillcolor="rgba(99,102,241,0.06)",
                  line=dict(width=1, color="rgba(99,102,241,0.3)"))
    fig.update_layout(**_base_layout(),
        title=dict(text=f"PSR {psr_name}  —  σ={sigma:.4f}{p_str}  |  log₁₀(BF)={log_bf:.2f}",
                   font=dict(size=14, color=COLORS["text"])),
        xaxis_title="Time relative to GW Event (s)", yaxis_title="Timing Residual (µs)",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"))
    return fig


def build_sweep_fig(sweeps):
    if not sweeps:
        return empty_fig("No sweep data. Click Run Audit first.")
    fig = go.Figure()
    for i, (psr, data) in enumerate(sweeps.items()):
        fig.add_trace(go.Scattergl(x=data["offsets"] / 86400.0, y=data["sigmas"], mode="lines",
            line=dict(color=PULSAR_COLORS[i % len(PULSAR_COLORS)], width=1.5),
            name=psr, opacity=0.7))
    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2),
                  annotation_text="GW Event", annotation_font_color=COLORS["red"])
    fig.add_hline(y=3.0, line=dict(color=COLORS["yellow"], width=1, dash="dash"),
                  annotation_text="3σ", annotation_font_color=COLORS["yellow"])
    fig.update_layout(**_base_layout(600),
        title=dict(text="Sliding Window Sweep — σ vs Time Offset", font=dict(size=16)),
        xaxis_title="Offset from GW Event (days)", yaxis_title="σ Deviation", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11)))
    return fig


def build_sky_map(ensemble, audit_results):
    if not ensemble:
        return empty_fig("No data.")
    names = list(ensemble.keys())
    lats, lons, sigmas, texts = [], [], [], []
    for psr in names:
        if psr in PULSAR_COORDS:
            ra, dec = PULSAR_COORDS[psr]
            lon = ra - 360 if ra > 180 else ra
            lats.append(dec); lons.append(lon)
            s = audit_results.get("pulsar_results", {}).get(psr, {}).get("sigma", 0)
            sigmas.append(abs(s))
            texts.append(f"{psr}<br>σ={s:.3f}")
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, text=texts, hoverinfo="text", mode="markers+text",
        textposition="top center", textfont=dict(size=10, color=COLORS["text"]),
        marker=dict(size=[max(10, s * 15) for s in sigmas],
                    color=sigmas, colorscale="Viridis", cmin=0, cmax=max(sigmas) if sigmas else 1,
                    colorbar=dict(title="σ", titlefont=dict(color=COLORS["text"]),
                                  tickfont=dict(color=COLORS["text"])),
                    line=dict(width=1, color="white"), opacity=0.9)))
    fig.update_geos(projection_type="mollweide", showland=False, showocean=False,
                    showframe=True, framecolor=COLORS["card_border"], framewidth=1,
                    bgcolor=COLORS["bg"], showcoastlines=False,
                    lonaxis=dict(gridcolor="#1e293b"), lataxis=dict(gridcolor="#1e293b"))
    fig.update_layout(**_base_layout(550), title=dict(text="Pulsar Sky Map (σ Deviation)", font=dict(size=16)))
    return fig


def build_correlation_fig(corr_data):
    if corr_data is None:
        return empty_fig("Run audit first.")
    names = corr_data["names"]
    matrix = corr_data["matrix"]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=names, y=names, colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(matrix, 3), texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="Pearson r", titlefont=dict(color=COLORS["text"]),
                      tickfont=dict(color=COLORS["text"]))))
    fig.update_layout(**_base_layout(600),
        title=dict(text="Pairwise Residual Correlation Matrix", font=dict(size=16)),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"))
    return fig


def build_null_dist_fig(null_data, observed_sigma):
    if null_data is None:
        return empty_fig("Run audit first.")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=null_data["null_sigmas"], nbinsx=30,
        marker_color=COLORS["accent"], opacity=0.7, name="Null Distribution"))
    fig.add_vline(x=observed_sigma, line=dict(color=COLORS["red"], width=3),
                  annotation_text=f"Observed σ = {observed_sigma:.3f}",
                  annotation_font_color=COLORS["red"])
    fig.add_vline(x=null_data["mean"], line=dict(color=COLORS["green"], width=2, dash="dash"),
                  annotation_text=f"Null μ = {null_data['mean']:.3f}",
                  annotation_font_color=COLORS["green"])
    fig.update_layout(**_base_layout(500),
        title=dict(text="Null Distribution Test — Ensemble σ at Random Times", font=dict(size=16)),
        xaxis_title="σ Deviation", yaxis_title="Count",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"))
    return fig


def build_hellings_downs_fig(hd_data):
    if hd_data is None or len(hd_data.get("angles_deg", [])) == 0:
        return empty_fig("Insufficient data for Hellings-Downs.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hd_data["hd_curve_x"], y=hd_data["hd_curve_y"],
        mode="lines", line=dict(color=COLORS["yellow"], width=2, dash="dash"),
        name="Theoretical HD Curve"))
    fig.add_trace(go.Scatter(x=hd_data["angles_deg"], y=hd_data["correlations"],
        mode="markers", marker=dict(size=10, color=COLORS["accent"],
            line=dict(width=1, color="white")), name="Measured Pairs"))
    fig.update_layout(**_base_layout(500),
        title=dict(text="Hellings-Downs Curve — Cross-Correlation vs Angular Separation", font=dict(size=15)),
        xaxis_title="Angular Separation (°)", yaxis_title="Cross-Correlation",
        xaxis=dict(gridcolor="#1e293b", range=[0, 180]),
        yaxis=dict(gridcolor="#1e293b", range=[-1, 1]),
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95))
    return fig
