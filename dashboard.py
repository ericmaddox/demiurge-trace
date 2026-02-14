"""
Demiurge Trace — Interactive Dashboard
=======================================
A Plotly Dash web application for exploring simulation hypothesis audit results.
Launch with: python dashboard.py
"""

import os
import sys
import numpy as np
from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading helpers (cached in module-level dict to avoid re-computation)
# ---------------------------------------------------------------------------
_PULSAR_CACHE = {}
_STRAIN_CACHE = {}

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


def load_ensemble(gw_time, simulate_lag=False):
    """Load all available pulsars. Results are cached by (gw_time, simulate_lag)."""
    cache_key = (gw_time, simulate_lag)
    if cache_key in _PULSAR_CACHE:
        return _PULSAR_CACHE[cache_key]

    from demiurge_trace import pulsar_module
    data = pulsar_module.get_ensemble_data(gw_time, is_simulation=simulate_lag)
    _PULSAR_CACHE[cache_key] = data
    return data


def load_strain(gw_time):
    """Load GW strain data. Cached by gw_time."""
    if gw_time in _STRAIN_CACHE:
        return _STRAIN_CACHE[gw_time]

    from demiurge_trace import ligo_module
    try:
        strain = ligo_module.fetch_strain_data(gw_time)
    except Exception:
        strain = None
    _STRAIN_CACHE[gw_time] = strain
    return strain


def run_audit(gw_time, ensemble_data, window_seconds):
    """Run the auditor on pre-loaded data."""
    from demiurge_trace import auditor
    return auditor.audit_ensemble(gw_time, ensemble_data, window_seconds=window_seconds)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
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


def _empty_fig(message="Press ▶ Run Audit to begin."):
    """Return an empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color=COLORS["text_dim"]))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout — ALL IDs exist in initial render (no dynamic ID creation)
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Demiurge Trace — Simulation Hypothesis Auditor"

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "color": COLORS["text"],
    },
    children=[
        # ── Header ──────────────────────────────────────────────────────
        html.Div(
            style={
                "background": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
                "borderBottom": f"1px solid {COLORS['card_border']}",
                "padding": "24px 40px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
            children=[
                html.Div([
                    html.H1(
                        "🔭 Demiurge Trace",
                        style={
                            "margin": "0",
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "background": "linear-gradient(90deg, #a78bfa, #6366f1, #818cf8)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                        },
                    ),
                    html.P(
                        "Simulation Hypothesis Auditor",
                        style={"margin": "4px 0 0 0", "color": COLORS["text_dim"], "fontSize": "14px"},
                    ),
                ]),
                html.Div(
                    id="status-badge",
                    children="AWAITING AUDIT",
                    style={
                        "padding": "8px 20px",
                        "borderRadius": "20px",
                        "fontSize": "13px",
                        "fontWeight": "600",
                        "letterSpacing": "0.5px",
                        "backgroundColor": "rgba(148,163,184,0.15)",
                        "color": COLORS["text_dim"],
                    },
                ),
            ],
        ),

        # ── Main content ────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "0", "minHeight": "calc(100vh - 90px)"},
            children=[
                # ── Sidebar ─────────────────────────────────────────────
                html.Div(
                    style={
                        "width": "300px",
                        "minWidth": "300px",
                        "backgroundColor": COLORS["card"],
                        "borderRight": f"1px solid {COLORS['card_border']}",
                        "padding": "28px 24px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "24px",
                    },
                    children=[
                        html.Div([
                            html.Label("Gravitational Wave Event", style={"fontSize": "12px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"], "marginBottom": "8px", "display": "block"}),
                            dcc.Dropdown(
                                id="event-dropdown",
                                options=[{"label": k, "value": k} for k in KNOWN_EVENTS],
                                value="GW170817",
                                clearable=False,
                                style={"backgroundColor": "#1e293b", "color": "#e2e8f0", "borderColor": "#334155", "borderRadius": "8px"},
                            ),
                        ]),
                        html.Div([
                            html.Label("Analysis Window", style={"fontSize": "12px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"], "marginBottom": "8px", "display": "block"}),
                            dcc.Dropdown(
                                id="window-dropdown",
                                options=[{"label": k, "value": v} for k, v in WINDOW_PRESETS.items()],
                                value=432000,
                                clearable=False,
                                style={"backgroundColor": "#1e293b", "color": "#e2e8f0", "borderColor": "#334155", "borderRadius": "8px"},
                            ),
                        ]),
                        html.Div([
                            html.Label("Simulation Mode", style={"fontSize": "12px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"], "marginBottom": "8px", "display": "block"}),
                            dcc.Checklist(
                                id="simulate-lag-check",
                                options=[{"label": " Inject 10µs lag spike", "value": "sim"}],
                                value=[],
                                style={"color": COLORS["text"], "fontSize": "14px"},
                                inputStyle={"marginRight": "8px"},
                            ),
                        ]),
                        html.Button(
                            "▶  Run Audit",
                            id="run-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": COLORS["accent"],
                                "color": "white",
                                "border": "none",
                                "padding": "14px 24px",
                                "borderRadius": "10px",
                                "fontSize": "15px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "transition": "all 0.2s",
                                "boxShadow": f"0 4px 20px {COLORS['accent_glow']}",
                                "marginTop": "8px",
                            },
                        ),
                        html.Div(id="loading-text", style={"color": COLORS["text_dim"], "fontSize": "13px", "textAlign": "center"}),

                        # Results card
                        html.Div(
                            id="results-card",
                            style={
                                "marginTop": "auto",
                                "backgroundColor": "rgba(99,102,241,0.08)",
                                "border": f"1px solid {COLORS['card_border']}",
                                "borderRadius": "12px",
                                "padding": "20px",
                            },
                            children=[html.P("Run an audit to see results.", style={"color": COLORS["text_dim"], "fontSize": "14px", "margin": "0"})],
                        ),
                    ],
                ),

                # ── Charts area ─────────────────────────────────────────
                html.Div(
                    style={"flex": "1", "padding": "28px 32px", "overflowY": "auto"},
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            value="tab-ensemble",
                            style={"marginBottom": "20px"},
                            colors={"border": COLORS["card_border"], "primary": COLORS["accent"], "background": COLORS["card"]},
                            children=[
                                dcc.Tab(
                                    label="Ensemble Overview",
                                    value="tab-ensemble",
                                    style={"backgroundColor": COLORS["card"], "color": COLORS["text_dim"], "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 20px"},
                                    selected_style={"backgroundColor": COLORS["accent"], "color": "white", "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 20px", "fontWeight": "600"},
                                ),
                                dcc.Tab(
                                    label="Per-Pulsar Detail",
                                    value="tab-pulsar",
                                    style={"backgroundColor": COLORS["card"], "color": COLORS["text_dim"], "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 20px"},
                                    selected_style={"backgroundColor": COLORS["accent"], "color": "white", "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 20px", "fontWeight": "600"},
                                ),
                                dcc.Tab(
                                    label="Results Table",
                                    value="tab-table",
                                    style={"backgroundColor": COLORS["card"], "color": COLORS["text_dim"], "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 20px"},
                                    selected_style={"backgroundColor": COLORS["accent"], "color": "white", "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 20px", "fontWeight": "600"},
                                ),
                            ],
                        ),

                        # --- ALL tab content panels exist in DOM from the start ---

                        # Tab 1: Ensemble
                        html.Div(
                            id="panel-ensemble",
                            children=[dcc.Graph(id="ensemble-graph", figure=_empty_fig(), config={"displayModeBar": True, "scrollZoom": True}, style={"borderRadius": "12px", "overflow": "hidden"})],
                        ),

                        # Tab 2: Per-Pulsar
                        html.Div(
                            id="panel-pulsar",
                            style={"display": "none"},
                            children=[
                                dcc.Dropdown(
                                    id="pulsar-select",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    style={"backgroundColor": "#1e293b", "color": "#e2e8f0", "borderColor": "#334155", "borderRadius": "8px", "marginBottom": "20px", "maxWidth": "400px"},
                                ),
                                dcc.Graph(id="pulsar-graph", figure=_empty_fig("Select a pulsar above."), config={"displayModeBar": True, "scrollZoom": True}, style={"borderRadius": "12px", "overflow": "hidden"}),
                            ],
                        ),

                        # Tab 3: Results Table
                        html.Div(
                            id="panel-table",
                            style={"display": "none"},
                            children=[
                                dash_table.DataTable(
                                    id="results-table",
                                    data=[],
                                    columns=[{"name": c, "id": c} for c in ["Pulsar", "σ Deviation", "Window RMS (s)", "Baseline RMS (s)", "Artifact"]],
                                    style_table={"overflowX": "auto", "borderRadius": "12px"},
                                    style_header={
                                        "backgroundColor": COLORS["card"],
                                        "color": COLORS["text"],
                                        "fontWeight": "600",
                                        "fontSize": "13px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px",
                                        "border": f"1px solid {COLORS['card_border']}",
                                        "padding": "12px 16px",
                                    },
                                    style_data={
                                        "backgroundColor": COLORS["bg"],
                                        "color": COLORS["text"],
                                        "fontSize": "14px",
                                        "fontFamily": "'JetBrains Mono', 'Cascadia Code', monospace",
                                        "border": f"1px solid {COLORS['card_border']}",
                                        "padding": "10px 16px",
                                    },
                                    style_data_conditional=[
                                        {"if": {"filter_query": '{Artifact} contains "YES"'}, "color": COLORS["red"], "fontWeight": "bold"},
                                        {"if": {"filter_query": '{Artifact} contains "No"'}, "color": COLORS["green"]},
                                    ],
                                    style_cell={"textAlign": "center"},
                                    page_size=15,
                                ),
                                html.Div(id="verdict-banner", style={"marginTop": "20px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Hidden stores
        dcc.Store(id="audit-data"),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ── Tab switching: show/hide panels ──────────────────────────────────────────
@callback(
    Output("panel-ensemble", "style"),
    Output("panel-pulsar", "style"),
    Output("panel-table", "style"),
    Input("tabs", "value"),
)
def switch_tab(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    if tab == "tab-ensemble":
        return show, hide, hide
    elif tab == "tab-pulsar":
        return hide, show, hide
    elif tab == "tab-table":
        return hide, hide, show
    return show, hide, hide


# ── Run audit ────────────────────────────────────────────────────────────────
@callback(
    Output("audit-data", "data"),
    Output("loading-text", "children"),
    Input("run-button", "n_clicks"),
    State("event-dropdown", "value"),
    State("window-dropdown", "value"),
    State("simulate-lag-check", "value"),
    prevent_initial_call=True,
)
def run_audit_callback(n_clicks, event_name, window, simulate_lag):
    """Run the full audit pipeline and store results."""
    if n_clicks == 0:
        return no_update, ""

    gw_time = KNOWN_EVENTS[event_name]
    sim = "sim" in (simulate_lag or [])

    ensemble = load_ensemble(gw_time, simulate_lag=sim)
    if not ensemble:
        return no_update, "❌ No pulsar data available."

    strain = load_strain(gw_time)
    results = run_audit(gw_time, ensemble, window)

    # Serialize for dcc.Store (must be JSON-serializable)
    serialized = {
        "event": event_name,
        "gw_time": gw_time,
        "window": window,
        "simulate_lag": sim,
        "ensemble_sigma": results["ensemble_sigma"],
        "n_pulsars_in_window": results["n_pulsars_in_window"],
        "n_detectors_above_1s": results["n_detectors_above_1s"],
        "is_ensemble_artifact": results["is_ensemble_artifact"],
        "pulsars": {},
    }

    for psr_name, (times, residuals) in ensemble.items():
        psr_result = results["pulsar_results"].get(psr_name, {})
        serialized["pulsars"][psr_name] = {
            "times": times.tolist(),
            "residuals": residuals.tolist(),
            "sigma": psr_result.get("sigma", 0),
            "window_rms": psr_result.get("window_rms", 0),
            "baseline_rms": psr_result.get("baseline_rms", 0),
            "is_artifact": psr_result.get("is_artifact", False),
        }

    if strain is not None:
        serialized["strain_times"] = (strain.times.value - gw_time).tolist()
        serialized["strain_values"] = strain.value.tolist()

    return serialized, f"✅ Loaded {len(ensemble)} pulsars."


# ── Status badge ─────────────────────────────────────────────────────────────
@callback(
    Output("status-badge", "children"),
    Output("status-badge", "style"),
    Input("audit-data", "data"),
)
def update_status_badge(data):
    base_style = {
        "padding": "8px 20px",
        "borderRadius": "20px",
        "fontSize": "13px",
        "fontWeight": "600",
        "letterSpacing": "0.5px",
    }
    if data is None:
        return "AWAITING AUDIT", {**base_style, "backgroundColor": "rgba(148,163,184,0.15)", "color": COLORS["text_dim"]}
    if data["is_ensemble_artifact"]:
        return "🚨 ARTIFACT DETECTED", {**base_style, "backgroundColor": "rgba(239,68,68,0.2)", "color": COLORS["red"], "border": f"1px solid {COLORS['red']}"}
    return "✓ NULL HYPOTHESIS", {**base_style, "backgroundColor": "rgba(16,185,129,0.15)", "color": COLORS["green"], "border": f"1px solid {COLORS['green']}"}


# ── Results card ─────────────────────────────────────────────────────────────
@callback(
    Output("results-card", "children"),
    Input("audit-data", "data"),
)
def update_results_card(data):
    if data is None:
        return html.P("Run an audit to see results.", style={"color": COLORS["text_dim"], "fontSize": "14px", "margin": "0"})

    sigma_color = COLORS["green"] if abs(data["ensemble_sigma"]) < 1 else (COLORS["yellow"] if abs(data["ensemble_sigma"]) < 3 else COLORS["red"])

    return [
        html.H3("Audit Results", style={"margin": "0 0 14px 0", "fontSize": "14px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"]}),
        _result_row("Event", data["event"]),
        _result_row("Window", f"{data['window'] / 86400:.1f} days"),
        _result_row("Pulsars in Window", str(data["n_pulsars_in_window"])),
        _result_row("Ensemble Sigma", f"{data['ensemble_sigma']:.4f}", sigma_color),
        _result_row("Detectors > 1σ", str(data["n_detectors_above_1s"])),
        html.Hr(style={"borderColor": COLORS["card_border"], "margin": "12px 0"}),
        _result_row(
            "Verdict",
            "ARTIFACT" if data["is_ensemble_artifact"] else "NULL",
            COLORS["red"] if data["is_ensemble_artifact"] else COLORS["green"],
        ),
    ]


def _result_row(label, value, color=None):
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px", "fontSize": "13px"},
        children=[
            html.Span(label, style={"color": COLORS["text_dim"]}),
            html.Span(value, style={"color": color or COLORS["text"], "fontWeight": "600", "fontFamily": "'JetBrains Mono', 'Cascadia Code', monospace"}),
        ],
    )


# ── Ensemble chart ───────────────────────────────────────────────────────────
@callback(
    Output("ensemble-graph", "figure"),
    Input("audit-data", "data"),
)
def update_ensemble_chart(data):
    if data is None:
        return _empty_fig()

    gw_time = data["gw_time"]
    has_strain = "strain_times" in data

    rows = 2 if has_strain else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.3, 0.7] if has_strain else [1],
        subplot_titles=["GW Strain (H1)", "Pulsar Timing Residuals (Normalized σ)"] if has_strain else ["Pulsar Timing Residuals (Normalized σ)"],
    )

    if has_strain:
        fig.add_trace(
            go.Scatter(
                x=data["strain_times"],
                y=data["strain_values"],
                mode="lines",
                line=dict(color=COLORS["strain"], width=1),
                name="GW Strain (H1)",
                hovertemplate="t=%{x:.3f}s<br>Strain=%{y:.2e}<extra>H1</extra>",
            ),
            row=1, col=1,
        )
        fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2), row=1, col=1)

    chart_row = 2 if has_strain else 1
    for i, (psr_name, psr_data) in enumerate(data["pulsars"].items()):
        times = np.array(psr_data["times"]) - gw_time
        residuals = np.array(psr_data["residuals"])
        std = np.std(residuals)
        norm = residuals / std if std > 0 else residuals
        color = PULSAR_COLORS[i % len(PULSAR_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=times.tolist(),
                y=norm.tolist(),
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.5),
                name=psr_name,
                hovertemplate=f"<b>{psr_name}</b><br>Δt=%{{x:.1f}}s<br>σ=%{{y:.2f}}<extra></extra>",
            ),
            row=chart_row, col=1,
        )

    window = data["window"]
    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2, dash="solid"), row=chart_row, col=1)
    fig.add_vrect(x0=-window, x1=window, fillcolor="rgba(245,158,11,0.06)", line=dict(width=1, color="rgba(245,158,11,0.3)"), row=chart_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02 if has_strain else 1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="closest",
    )

    fig.update_xaxes(title_text="Time relative to GW Event (s)", row=chart_row, col=1, gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")

    return fig


# ── Pulsar dropdown options ──────────────────────────────────────────────────
@callback(
    Output("pulsar-select", "options"),
    Output("pulsar-select", "value"),
    Input("audit-data", "data"),
)
def update_pulsar_dropdown(data):
    if data is None:
        return [], None
    psr_names = list(data["pulsars"].keys())
    return [{"label": p, "value": p} for p in psr_names], psr_names[0] if psr_names else None


# ── Pulsar detail chart ──────────────────────────────────────────────────────
@callback(
    Output("pulsar-graph", "figure"),
    Input("pulsar-select", "value"),
    Input("audit-data", "data"),
)
def update_pulsar_chart(psr_name, data):
    if data is None or psr_name is None or psr_name not in data.get("pulsars", {}):
        return _empty_fig("Select a pulsar above.")

    psr = data["pulsars"][psr_name]
    gw_time = data["gw_time"]
    window = data["window"]

    times = np.array(psr["times"]) - gw_time
    residuals = np.array(psr["residuals"]) * 1e6  # µs

    mask = (times >= -window) & (times <= window)

    fig = go.Figure()

    # Baseline residuals
    if np.any(~mask):
        fig.add_trace(go.Scatter(
            x=times[~mask].tolist(),
            y=residuals[~mask].tolist(),
            mode="markers",
            marker=dict(size=4, color=COLORS["text_dim"], opacity=0.3),
            name="Baseline",
            hovertemplate="Δt=%{x:.1f}s<br>Residual=%{y:.3f} µs<extra>Baseline</extra>",
        ))

    # In-window residuals
    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=times[mask].tolist(),
            y=residuals[mask].tolist(),
            mode="markers",
            marker=dict(size=6, color=COLORS["accent"], opacity=0.8, line=dict(width=1, color="white")),
            name="In Window",
            hovertemplate="Δt=%{x:.1f}s<br>Residual=%{y:.3f} µs<extra>Window</extra>",
        ))

    fig.add_vline(x=0, line=dict(color=COLORS["red"], width=2))
    fig.add_vrect(x0=-window, x1=window, fillcolor="rgba(99,102,241,0.06)", line=dict(width=1, color="rgba(99,102,241,0.3)"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        title=dict(
            text=f"PSR {psr_name}  —  σ = {psr['sigma']:.4f}  |  Window RMS = {psr['window_rms']:.2e} s  |  Baseline RMS = {psr['baseline_rms']:.2e} s",
            font=dict(size=14, color=COLORS["text"]),
        ),
        xaxis_title="Time relative to GW Event (s)",
        yaxis_title="Timing Residual (µs)",
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
        height=500,
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="closest",
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    )

    return fig


# ── Results table ────────────────────────────────────────────────────────────
@callback(
    Output("results-table", "data"),
    Output("verdict-banner", "children"),
    Input("audit-data", "data"),
)
def update_results_table(data):
    if data is None:
        return [], html.P("Run an audit to see results.", style={"color": COLORS["text_dim"]})

    rows = []
    for psr_name, psr_data in data["pulsars"].items():
        rows.append({
            "Pulsar": psr_name,
            "σ Deviation": f"{psr_data['sigma']:.4f}",
            "Window RMS (s)": f"{psr_data['window_rms']:.2e}",
            "Baseline RMS (s)": f"{psr_data['baseline_rms']:.2e}",
            "Artifact": "🚨 YES" if psr_data["is_artifact"] else "✓ No",
        })

    banner = html.Div(
        style={"padding": "20px", "backgroundColor": "rgba(99,102,241,0.08)", "borderRadius": "12px", "border": f"1px solid {COLORS['card_border']}"},
        children=[
            html.H3(
                f"Ensemble Verdict: {'COHERENT ARTIFACT DETECTED' if data['is_ensemble_artifact'] else 'Universal Clock Stability Confirmed'}",
                style={
                    "color": COLORS["red"] if data["is_ensemble_artifact"] else COLORS["green"],
                    "margin": "0 0 8px 0",
                    "fontSize": "18px",
                },
            ),
            html.P(
                f"Mean Ensemble σ = {data['ensemble_sigma']:.4f}  |  {data['n_pulsars_in_window']} pulsars in window  |  {data['n_detectors_above_1s']} above 1σ  |  Simulation: {'ON' if data['simulate_lag'] else 'OFF'}",
                style={"color": COLORS["text_dim"], "margin": "0", "fontSize": "14px", "fontFamily": "'JetBrains Mono', monospace"},
            ),
        ],
    )

    return rows, banner


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🔭 Demiurge Trace — Interactive Dashboard")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
