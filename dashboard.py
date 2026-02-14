"""
Demiurge Trace — Interactive Dashboard
=======================================
A Plotly Dash web application for exploring simulation hypothesis audit results.
Launch with: python dashboard.py
"""

import os
import sys
import time
import webbrowser
import threading
import numpy as np
from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Server-side data store (never serialized to the browser)
# ---------------------------------------------------------------------------
_ENSEMBLE_CACHE = {}   # (gw_time, sim) -> dict of {psr_name: (times, residuals)}
_STRAIN_CACHE = {}     # gw_time -> strain object or None
_AUDIT_RESULTS = {}    # session key -> small results dict

KNOWN_EVENTS = {
    "GW150914": 1126259462.4,
    "GW170817": 1187008882.4,
}

WINDOW_PRESETS = {
    "1 hour": 3600,
    "6 hours": 21600,
    "1 day": 86400,
    "3 days": 259200,
    "5 days (recommended)": 432000,
    "10 days": 864000,
}

COLORS = {
    "bg": "#0a0e17",
    "card": "#111827",
    "card_border": "#1e293b",
    "accent": "#6366f1",
    "accent_glow": "rgba(99,102,241,0.15)",
    "text": "#e2e8f0",
    "text_dim": "#94a3b8",
    "green": "#10b981",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "purple": "#a78bfa",
    "strain": "#c084fc",
}

PULSAR_COLORS = [
    "#6366f1", "#ec4899", "#14b8a6", "#f59e0b", "#ef4444",
    "#8b5cf6", "#06b6d4", "#84cc16", "#f97316", "#64748b",
]


def _load_ensemble(gw_time, simulate_lag=False):
    key = (gw_time, simulate_lag)
    if key not in _ENSEMBLE_CACHE:
        from demiurge_trace import pulsar_module
        _ENSEMBLE_CACHE[key] = pulsar_module.get_ensemble_data(gw_time, is_simulation=simulate_lag)
    return _ENSEMBLE_CACHE[key]


def _load_strain(gw_time):
    if gw_time not in _STRAIN_CACHE:
        from demiurge_trace import ligo_module
        try:
            _STRAIN_CACHE[gw_time] = ligo_module.fetch_strain_data(gw_time)
        except Exception:
            _STRAIN_CACHE[gw_time] = None
    return _STRAIN_CACHE[gw_time]


def _run_audit(gw_time, ensemble, window):
    from demiurge_trace import auditor
    return auditor.audit_ensemble(gw_time, ensemble, window_seconds=window)


def _empty_fig(msg="Press ▶ Run Audit to begin."):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color=COLORS["text_dim"]))
    fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                      plot_bgcolor=COLORS["bg"], height=500,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _result_row(label, value, color=None):
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px", "fontSize": "13px"},
        children=[
            html.Span(label, style={"color": COLORS["text_dim"]}),
            html.Span(value, style={"color": color or COLORS["text"], "fontWeight": "600",
                                     "fontFamily": "'JetBrains Mono', 'Cascadia Code', monospace"}),
        ],
    )


# ---------------------------------------------------------------------------
# Build figures directly from server-side cache (no serialization)
# ---------------------------------------------------------------------------

def _build_ensemble_fig(event_name, window):
    gw_time = KNOWN_EVENTS[event_name]
    ensemble = _ENSEMBLE_CACHE.get((gw_time, False), {})
    strain = _STRAIN_CACHE.get(gw_time)

    if not ensemble:
        return _empty_fig("No data loaded.")

    has_strain = strain is not None
    rows = 2 if has_strain else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.3, 0.7] if has_strain else [1],
        subplot_titles=(["GW Strain (H1)", "Pulsar Timing Residuals (Normalized σ)"]
                        if has_strain else ["Pulsar Timing Residuals (Normalized σ)"]),
    )

    if has_strain:
        st = strain.times.value - gw_time
        fig.add_trace(go.Scattergl(x=st, y=strain.value, mode="lines",
                                   line=dict(color=COLORS["strain"], width=1),
                                   name="GW Strain (H1)"), row=1, col=1)
        fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2), row=1, col=1)

    cr = 2 if has_strain else 1
    for i, (psr, (times, res)) in enumerate(ensemble.items()):
        t = times - gw_time
        std = np.std(res)
        norm = (res / std) if std > 0 else res
        fig.add_trace(go.Scattergl(
            x=t, y=norm, mode="markers",
            marker=dict(size=3, color=PULSAR_COLORS[i % len(PULSAR_COLORS)], opacity=0.5),
            name=psr,
            hovertemplate=f"<b>{psr}</b><br>Δt=%{{x:.1f}}s<br>σ=%{{y:.2f}}<extra></extra>",
        ), row=cr, col=1)

    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2), row=cr, col=1)
    fig.add_vrect(x0=-window, x1=window, fillcolor="rgba(245,158,11,0.06)",
                  line=dict(width=1, color="rgba(245,158,11,0.3)"), row=cr, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02 if has_strain else 1.05,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=30, t=60, b=40), hovermode="closest",
    )
    fig.update_xaxes(title_text="Time relative to GW Event (s)", row=cr, col=1,
                     gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    return fig


def _build_pulsar_fig(event_name, psr_name, window, audit_results):
    gw_time = KNOWN_EVENTS[event_name]
    ensemble = _ENSEMBLE_CACHE.get((gw_time, False), {})

    if psr_name not in ensemble:
        return _empty_fig("Pulsar not found.")

    times, residuals = ensemble[psr_name]
    t = times - gw_time
    res_us = residuals * 1e6  # microseconds
    mask = (t >= -window) & (t <= window)

    psr_res = audit_results.get("pulsar_results", {}).get(psr_name, {})
    sigma = psr_res.get("sigma", 0)
    w_rms = psr_res.get("window_rms", 0)
    b_rms = psr_res.get("baseline_rms", 0)

    fig = go.Figure()
    if np.any(~mask):
        fig.add_trace(go.Scattergl(
            x=t[~mask], y=res_us[~mask], mode="markers",
            marker=dict(size=4, color=COLORS["text_dim"], opacity=0.3), name="Baseline",
            hovertemplate="Δt=%{x:.1f}s<br>%{y:.3f} µs<extra>Baseline</extra>",
        ))
    if np.any(mask):
        fig.add_trace(go.Scattergl(
            x=t[mask], y=res_us[mask], mode="markers",
            marker=dict(size=6, color=COLORS["accent"], opacity=0.8,
                        line=dict(width=1, color="white")), name="In Window",
            hovertemplate="Δt=%{x:.1f}s<br>%{y:.3f} µs<extra>Window</extra>",
        ))

    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2))
    fig.add_vrect(x0=-window, x1=window, fillcolor="rgba(99,102,241,0.06)",
                  line=dict(width=1, color="rgba(99,102,241,0.3)"))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        title=dict(text=f"PSR {psr_name}  —  σ = {sigma:.4f}  |  Window RMS = {w_rms:.2e} s  |  Baseline RMS = {b_rms:.2e} s",
                   font=dict(size=14, color=COLORS["text"])),
        xaxis_title="Time relative to GW Event (s)", yaxis_title="Timing Residual (µs)",
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
        height=500, margin=dict(l=60, r=30, t=60, b=40), hovermode="closest",
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Demiurge Trace — Simulation Hypothesis Auditor"

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', 'Segoe UI', system-ui, sans-serif", "color": COLORS["text"]},
    children=[
        # Header
        html.Div(
            style={"background": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
                   "borderBottom": f"1px solid {COLORS['card_border']}", "padding": "24px 40px",
                   "display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            children=[
                html.Div([
                    html.H1("🔭 Demiurge Trace", style={
                        "margin": "0", "fontSize": "28px", "fontWeight": "700",
                        "background": "linear-gradient(90deg, #a78bfa, #6366f1, #818cf8)",
                        "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
                    html.P("Simulation Hypothesis Auditor",
                           style={"margin": "4px 0 0 0", "color": COLORS["text_dim"], "fontSize": "14px"}),
                ]),
                html.Div(id="status-badge", children="AWAITING AUDIT",
                         style={"padding": "8px 20px", "borderRadius": "20px", "fontSize": "13px",
                                "fontWeight": "600", "letterSpacing": "0.5px",
                                "backgroundColor": "rgba(148,163,184,0.15)", "color": COLORS["text_dim"]}),
            ],
        ),

        # Main
        html.Div(
            style={"display": "flex", "gap": "0", "minHeight": "calc(100vh - 90px)"},
            children=[
                # Sidebar
                html.Div(
                    style={"width": "300px", "minWidth": "300px", "backgroundColor": COLORS["card"],
                           "borderRight": f"1px solid {COLORS['card_border']}", "padding": "28px 24px",
                           "display": "flex", "flexDirection": "column", "gap": "24px"},
                    children=[
                        html.Div([
                            html.Label("Gravitational Wave Event", style={"fontSize": "12px", "fontWeight": "600",
                                       "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"],
                                       "marginBottom": "8px", "display": "block"}),
                            dcc.Dropdown(id="event-dropdown",
                                         options=[{"label": k, "value": k} for k in KNOWN_EVENTS],
                                         value="GW170817", clearable=False,
                                         style={"backgroundColor": "#1e293b", "borderRadius": "8px"}),
                        ]),
                        html.Div([
                            html.Label("Analysis Window", style={"fontSize": "12px", "fontWeight": "600",
                                       "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"],
                                       "marginBottom": "8px", "display": "block"}),
                            dcc.Dropdown(id="window-dropdown",
                                         options=[{"label": k, "value": v} for k, v in WINDOW_PRESETS.items()],
                                         value=432000, clearable=False,
                                         style={"backgroundColor": "#1e293b", "borderRadius": "8px"}),
                        ]),
                        html.Div([
                            html.Label("Simulation Mode", style={"fontSize": "12px", "fontWeight": "600",
                                       "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"],
                                       "marginBottom": "8px", "display": "block"}),
                            dcc.Checklist(id="simulate-lag-check",
                                          options=[{"label": " Inject 10µs lag spike", "value": "sim"}],
                                          value=[], style={"color": COLORS["text"], "fontSize": "14px"},
                                          inputStyle={"marginRight": "8px"}),
                        ]),
                        html.Button("▶  Run Audit", id="run-button", n_clicks=0,
                                    style={"backgroundColor": COLORS["accent"], "color": "white", "border": "none",
                                           "padding": "14px 24px", "borderRadius": "10px", "fontSize": "15px",
                                           "fontWeight": "600", "cursor": "pointer",
                                           "boxShadow": f"0 4px 20px {COLORS['accent_glow']}", "marginTop": "8px"}),
                        html.Div(id="loading-text", style={"color": COLORS["text_dim"], "fontSize": "13px", "textAlign": "center"}),
                        html.Div(id="results-card",
                                 style={"marginTop": "auto", "backgroundColor": "rgba(99,102,241,0.08)",
                                        "border": f"1px solid {COLORS['card_border']}", "borderRadius": "12px", "padding": "20px"},
                                 children=[html.P("Run an audit to see results.",
                                                  style={"color": COLORS["text_dim"], "fontSize": "14px", "margin": "0"})]),
                    ],
                ),

                # Charts
                html.Div(
                    style={"flex": "1", "padding": "28px 32px", "overflowY": "auto"},
                    children=[
                        dcc.Tabs(id="tabs", value="tab-ensemble", style={"marginBottom": "20px"},
                                 colors={"border": COLORS["card_border"], "primary": COLORS["accent"], "background": COLORS["card"]},
                                 children=[
                                     dcc.Tab(label="Ensemble Overview", value="tab-ensemble",
                                             style={"backgroundColor": COLORS["card"], "color": COLORS["text_dim"],
                                                    "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 20px"},
                                             selected_style={"backgroundColor": COLORS["accent"], "color": "white",
                                                             "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 20px", "fontWeight": "600"}),
                                     dcc.Tab(label="Per-Pulsar Detail", value="tab-pulsar",
                                             style={"backgroundColor": COLORS["card"], "color": COLORS["text_dim"],
                                                    "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 20px"},
                                             selected_style={"backgroundColor": COLORS["accent"], "color": "white",
                                                             "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 20px", "fontWeight": "600"}),
                                     dcc.Tab(label="Results Table", value="tab-table",
                                             style={"backgroundColor": COLORS["card"], "color": COLORS["text_dim"],
                                                    "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 20px"},
                                             selected_style={"backgroundColor": COLORS["accent"], "color": "white",
                                                             "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 20px", "fontWeight": "600"}),
                                 ]),
                        # Ensemble panel
                        html.Div(id="panel-ensemble",
                                 children=[dcc.Graph(id="ensemble-graph", figure=_empty_fig(),
                                                     config={"displayModeBar": True, "scrollZoom": True},
                                                     style={"borderRadius": "12px", "overflow": "hidden"})]),
                        # Pulsar panel
                        html.Div(id="panel-pulsar", style={"display": "none"},
                                 children=[
                                     dcc.Dropdown(id="pulsar-select", options=[], value=None, clearable=False,
                                                  style={"backgroundColor": "#1e293b", "borderRadius": "8px",
                                                         "marginBottom": "20px", "maxWidth": "400px"}),
                                     dcc.Graph(id="pulsar-graph", figure=_empty_fig("Select a pulsar."),
                                               config={"displayModeBar": True, "scrollZoom": True},
                                               style={"borderRadius": "12px", "overflow": "hidden"}),
                                 ]),
                        # Table panel
                        html.Div(id="panel-table", style={"display": "none"},
                                 children=[
                                     dash_table.DataTable(
                                         id="results-table", data=[], page_size=15,
                                         columns=[{"name": c, "id": c} for c in
                                                  ["Pulsar", "σ Deviation", "Window RMS (s)", "Baseline RMS (s)", "Artifact"]],
                                         style_table={"overflowX": "auto", "borderRadius": "12px"},
                                         style_header={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                                                       "fontWeight": "600", "fontSize": "13px", "textTransform": "uppercase",
                                                       "letterSpacing": "0.5px", "border": f"1px solid {COLORS['card_border']}",
                                                       "padding": "12px 16px"},
                                         style_data={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                                     "fontSize": "14px", "fontFamily": "'JetBrains Mono', monospace",
                                                     "border": f"1px solid {COLORS['card_border']}", "padding": "10px 16px"},
                                         style_data_conditional=[
                                             {"if": {"filter_query": '{Artifact} contains "YES"'}, "color": COLORS["red"], "fontWeight": "bold"},
                                             {"if": {"filter_query": '{Artifact} contains "No"'}, "color": COLORS["green"]},
                                         ],
                                         style_cell={"textAlign": "center"},
                                     ),
                                     html.Div(id="verdict-banner", style={"marginTop": "20px"}),
                                 ]),
                    ],
                ),
            ],
        ),

        # Only store a tiny config dict — NOT the raw data
        dcc.Store(id="audit-config", data=None),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks — all figure-building uses server-side cache directly
# ---------------------------------------------------------------------------

@callback(
    Output("panel-ensemble", "style"),
    Output("panel-pulsar", "style"),
    Output("panel-table", "style"),
    Input("tabs", "value"),
)
def switch_tab(tab):
    s, h = {"display": "block"}, {"display": "none"}
    return (s, h, h) if tab == "tab-ensemble" else (h, s, h) if tab == "tab-pulsar" else (h, h, s)


@callback(
    Output("audit-config", "data"),
    Output("loading-text", "children"),
    Input("run-button", "n_clicks"),
    State("event-dropdown", "value"),
    State("window-dropdown", "value"),
    State("simulate-lag-check", "value"),
    prevent_initial_call=True,
)
def run_audit_cb(n_clicks, event, window, sim_check):
    if n_clicks == 0:
        return no_update, ""

    gw_time = KNOWN_EVENTS[event]
    sim = "sim" in (sim_check or [])

    ensemble = _load_ensemble(gw_time, simulate_lag=sim)
    if not ensemble:
        return no_update, "❌ No pulsar data available."

    _load_strain(gw_time)
    results = _run_audit(gw_time, ensemble, window)

    # Store results server-side
    config_key = f"{event}_{window}_{sim}"
    _AUDIT_RESULTS[config_key] = results

    # Only pass a tiny config dict to the browser
    config = {
        "event": event,
        "window": window,
        "sim": sim,
        "key": config_key,
        "ensemble_sigma": results["ensemble_sigma"],
        "n_pulsars_in_window": results["n_pulsars_in_window"],
        "n_detectors_above_1s": results["n_detectors_above_1s"],
        "is_ensemble_artifact": results["is_ensemble_artifact"],
        "pulsar_names": list(ensemble.keys()),
        # Small per-pulsar summary (just scalars, not arrays)
        "pulsar_summaries": {
            psr: {
                "sigma": results["pulsar_results"].get(psr, {}).get("sigma", 0),
                "window_rms": results["pulsar_results"].get(psr, {}).get("window_rms", 0),
                "baseline_rms": results["pulsar_results"].get(psr, {}).get("baseline_rms", 0),
                "is_artifact": results["pulsar_results"].get(psr, {}).get("is_artifact", False),
            }
            for psr in ensemble.keys()
        },
    }

    return config, f"✅ Loaded {len(ensemble)} pulsars."


@callback(
    Output("status-badge", "children"),
    Output("status-badge", "style"),
    Input("audit-config", "data"),
)
def update_badge(cfg):
    base = {"padding": "8px 20px", "borderRadius": "20px", "fontSize": "13px",
            "fontWeight": "600", "letterSpacing": "0.5px"}
    if cfg is None:
        return "AWAITING AUDIT", {**base, "backgroundColor": "rgba(148,163,184,0.15)", "color": COLORS["text_dim"]}
    if cfg["is_ensemble_artifact"]:
        return "🚨 ARTIFACT DETECTED", {**base, "backgroundColor": "rgba(239,68,68,0.2)",
                                         "color": COLORS["red"], "border": f"1px solid {COLORS['red']}"}
    return "✓ NULL HYPOTHESIS", {**base, "backgroundColor": "rgba(16,185,129,0.15)",
                                  "color": COLORS["green"], "border": f"1px solid {COLORS['green']}"}


@callback(Output("results-card", "children"), Input("audit-config", "data"))
def update_card(cfg):
    if cfg is None:
        return html.P("Run an audit to see results.", style={"color": COLORS["text_dim"], "fontSize": "14px", "margin": "0"})
    sc = COLORS["green"] if abs(cfg["ensemble_sigma"]) < 1 else (COLORS["yellow"] if abs(cfg["ensemble_sigma"]) < 3 else COLORS["red"])
    return [
        html.H3("Audit Results", style={"margin": "0 0 14px 0", "fontSize": "14px", "fontWeight": "600",
                 "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"]}),
        _result_row("Event", cfg["event"]),
        _result_row("Window", f"{cfg['window'] / 86400:.1f} days"),
        _result_row("Pulsars in Window", str(cfg["n_pulsars_in_window"])),
        _result_row("Ensemble Sigma", f"{cfg['ensemble_sigma']:.4f}", sc),
        _result_row("Detectors > 1σ", str(cfg["n_detectors_above_1s"])),
        html.Hr(style={"borderColor": COLORS["card_border"], "margin": "12px 0"}),
        _result_row("Verdict", "ARTIFACT" if cfg["is_ensemble_artifact"] else "NULL",
                     COLORS["red"] if cfg["is_ensemble_artifact"] else COLORS["green"]),
    ]


@callback(Output("ensemble-graph", "figure"), Input("audit-config", "data"))
def update_ensemble(cfg):
    if cfg is None:
        return _empty_fig()
    return _build_ensemble_fig(cfg["event"], cfg["window"])


@callback(
    Output("pulsar-select", "options"),
    Output("pulsar-select", "value"),
    Input("audit-config", "data"),
)
def update_psr_dropdown(cfg):
    if cfg is None:
        return [], None
    names = cfg["pulsar_names"]
    return [{"label": p, "value": p} for p in names], names[0] if names else None


@callback(
    Output("pulsar-graph", "figure"),
    Input("pulsar-select", "value"),
    Input("audit-config", "data"),
)
def update_psr_chart(psr_name, cfg):
    if cfg is None or psr_name is None:
        return _empty_fig("Select a pulsar.")
    results = _AUDIT_RESULTS.get(cfg["key"], {})
    return _build_pulsar_fig(cfg["event"], psr_name, cfg["window"], results)


@callback(
    Output("results-table", "data"),
    Output("verdict-banner", "children"),
    Input("audit-config", "data"),
)
def update_table(cfg):
    if cfg is None:
        return [], html.P("Run an audit to see results.", style={"color": COLORS["text_dim"]})

    rows = []
    for psr, s in cfg["pulsar_summaries"].items():
        rows.append({
            "Pulsar": psr,
            "σ Deviation": f"{s['sigma']:.4f}",
            "Window RMS (s)": f"{s['window_rms']:.2e}",
            "Baseline RMS (s)": f"{s['baseline_rms']:.2e}",
            "Artifact": "🚨 YES" if s["is_artifact"] else "✓ No",
        })

    banner = html.Div(
        style={"padding": "20px", "backgroundColor": "rgba(99,102,241,0.08)", "borderRadius": "12px",
               "border": f"1px solid {COLORS['card_border']}"},
        children=[
            html.H3(
                f"Ensemble Verdict: {'COHERENT ARTIFACT DETECTED' if cfg['is_ensemble_artifact'] else 'Universal Clock Stability Confirmed'}",
                style={"color": COLORS["red"] if cfg["is_ensemble_artifact"] else COLORS["green"],
                       "margin": "0 0 8px 0", "fontSize": "18px"}),
            html.P(
                f"Mean Ensemble σ = {cfg['ensemble_sigma']:.4f}  |  {cfg['n_pulsars_in_window']} pulsars  |  "
                f"{cfg['n_detectors_above_1s']} above 1σ  |  Sim: {'ON' if cfg['sim'] else 'OFF'}",
                style={"color": COLORS["text_dim"], "margin": "0", "fontSize": "14px",
                       "fontFamily": "'JetBrains Mono', monospace"}),
        ],
    )
    return rows, banner


# ---------------------------------------------------------------------------
# Pre-load data and start server
# ---------------------------------------------------------------------------
def _preload():
    t0 = time.time()
    for name, gps in KNOWN_EVENTS.items():
        print(f"  Loading ensemble for {name}...")
        try:
            _load_ensemble(gps, simulate_lag=False)
        except Exception as e:
            print(f"    ⚠ {e}")
        print(f"  Loading strain for {name}...")
        try:
            _load_strain(gps)
        except Exception as e:
            print(f"    ⚠ Strain unavailable: {e}")
    print(f"\n  ✅ Pre-loaded in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🔭 Demiurge Trace — Interactive Dashboard")
    print("=" * 60)
    print("\n  Pre-loading data (this may take a minute)...\n")
    _preload()
    print("  Dashboard ready at http://127.0.0.1:8050\n")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    app.run(debug=False, host="127.0.0.1", port=8050)
