"""
Demiurge Trace — Interactive Dashboard
=======================================
Launch with: python dashboard.py
"""
import os, sys, time, webbrowser, threading, json, io, csv
import numpy as np
from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, no_update
import warnings
warnings.filterwarnings("ignore")

from dashboard_figures import (
    COLORS, PULSAR_COLORS, PULSAR_COORDS, empty_fig,
    build_ensemble_fig, build_pulsar_fig, build_sweep_fig,
    build_sky_map, build_correlation_fig, build_null_dist_fig,
    build_hellings_downs_fig,
)

# ---------------------------------------------------------------------------
# Server-side data caches
# ---------------------------------------------------------------------------
_ENSEMBLE_CACHE = {}
_STRAIN_CACHE = {}
_AUDIT_RESULTS = {}
_SWEEP_CACHE = {}
_CORR_CACHE = {}
_NULL_CACHE = {}
_HD_CACHE = {}

KNOWN_EVENTS = {"GW150914": 1126259462.4, "GW170817": 1187008882.4}
WINDOW_PRESETS = {
    "1 hour": 3600, "6 hours": 21600, "1 day": 86400,
    "3 days": 259200, "5 days (recommended)": 432000, "10 days": 864000,
}


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


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
TAB_STYLE = {"backgroundColor": COLORS["card"], "color": COLORS["text_dim"],
             "border": f"1px solid {COLORS['card_border']}", "borderRadius": "8px 8px 0 0", "padding": "10px 16px"}
TAB_SEL = {"backgroundColor": COLORS["accent"], "color": "white",
           "border": "none", "borderRadius": "8px 8px 0 0", "padding": "10px 16px", "fontWeight": "600"}


def _result_row(label, value, color=None):
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px", "fontSize": "13px"},
        children=[html.Span(label, style={"color": COLORS["text_dim"]}),
                  html.Span(value, style={"color": color or COLORS["text"], "fontWeight": "600",
                                           "fontFamily": "'JetBrains Mono', monospace"})])


def _make_graph(gid, msg=None):
    return dcc.Graph(id=gid, figure=empty_fig(msg or "Press ▶ Run Audit to begin."),
                     config={"displayModeBar": True, "scrollZoom": True},
                     style={"borderRadius": "12px", "overflow": "hidden"})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Demiurge Trace — Simulation Hypothesis Auditor"

TAB_DEFS = [
    ("Ensemble", "tab-ensemble"), ("Per-Pulsar", "tab-pulsar"),
    ("Sliding Window", "tab-sweep"), ("Sky Map", "tab-skymap"),
    ("Correlations", "tab-corr"), ("Null Distribution", "tab-null"),
    ("Hellings-Downs", "tab-hd"), ("Results", "tab-table"),
]
PANEL_IDS = [f"panel-{t[1].replace('tab-','')}" for t in TAB_DEFS]

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', system-ui, sans-serif", "color": COLORS["text"]},
    children=[
        # Header
        html.Div(style={"background": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
                         "borderBottom": f"1px solid {COLORS['card_border']}", "padding": "24px 40px",
                         "display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                 children=[
                     html.Div([
                         html.H1("🔭 Demiurge Trace", style={"margin": "0", "fontSize": "28px", "fontWeight": "700",
                             "background": "linear-gradient(90deg, #a78bfa, #6366f1, #818cf8)",
                             "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
                         html.P("Simulation Hypothesis Auditor",
                                style={"margin": "4px 0 0 0", "color": COLORS["text_dim"], "fontSize": "14px"}),
                     ]),
                     html.Div(id="status-badge", children="AWAITING AUDIT",
                              style={"padding": "8px 20px", "borderRadius": "20px", "fontSize": "13px",
                                     "fontWeight": "600", "letterSpacing": "0.5px",
                                     "backgroundColor": "rgba(148,163,184,0.15)", "color": COLORS["text_dim"]}),
                 ]),
        # Body
        html.Div(style={"display": "flex", "gap": "0", "minHeight": "calc(100vh - 90px)"},
                 children=[
            # Sidebar
            html.Div(style={"width": "300px", "minWidth": "300px", "backgroundColor": COLORS["card"],
                             "borderRight": f"1px solid {COLORS['card_border']}", "padding": "28px 24px",
                             "display": "flex", "flexDirection": "column", "gap": "20px"},
                     children=[
                html.Div([html.Label("GW Event", style={"fontSize": "11px", "fontWeight": "600",
                           "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"],
                           "marginBottom": "6px", "display": "block"}),
                          dcc.Dropdown(id="event-dropdown",
                                       options=[{"label": k, "value": k} for k in KNOWN_EVENTS],
                                       value="GW170817", clearable=False,
                                       style={"backgroundColor": "#1e293b", "borderRadius": "8px"})]),
                html.Div([html.Label("Window", style={"fontSize": "11px", "fontWeight": "600",
                           "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"],
                           "marginBottom": "6px", "display": "block"}),
                          dcc.Dropdown(id="window-dropdown",
                                       options=[{"label": k, "value": v} for k, v in WINDOW_PRESETS.items()],
                                       value=432000, clearable=False,
                                       style={"backgroundColor": "#1e293b", "borderRadius": "8px"})]),
                html.Div([html.Label("Options", style={"fontSize": "11px", "fontWeight": "600",
                           "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"],
                           "marginBottom": "6px", "display": "block"}),
                          dcc.Checklist(id="simulate-lag-check",
                                        options=[{"label": " Inject 10µs lag", "value": "sim"}],
                                        value=[], style={"color": COLORS["text"], "fontSize": "13px"},
                                        inputStyle={"marginRight": "8px"})]),
                html.Button("▶  Run Audit", id="run-button", n_clicks=0,
                            style={"backgroundColor": COLORS["accent"], "color": "white", "border": "none",
                                   "padding": "14px", "borderRadius": "10px", "fontSize": "15px",
                                   "fontWeight": "600", "cursor": "pointer",
                                   "boxShadow": f"0 4px 20px {COLORS['accent_glow']}"}),
                html.Div(id="loading-text", style={"color": COLORS["text_dim"], "fontSize": "12px", "textAlign": "center"}),
                # Export buttons
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    html.Button("📄 CSV", id="btn-csv", n_clicks=0,
                                style={"flex": "1", "padding": "8px", "borderRadius": "8px", "fontSize": "12px",
                                       "fontWeight": "600", "cursor": "pointer", "border": f"1px solid {COLORS['card_border']}",
                                       "backgroundColor": COLORS["card"], "color": COLORS["text"]}),
                    html.Button("📋 JSON", id="btn-json", n_clicks=0,
                                style={"flex": "1", "padding": "8px", "borderRadius": "8px", "fontSize": "12px",
                                       "fontWeight": "600", "cursor": "pointer", "border": f"1px solid {COLORS['card_border']}",
                                       "backgroundColor": COLORS["card"], "color": COLORS["text"]}),
                ]),
                dcc.Download(id="download-data"),
                html.Div(id="results-card",
                         style={"marginTop": "auto", "backgroundColor": "rgba(99,102,241,0.08)",
                                "border": f"1px solid {COLORS['card_border']}", "borderRadius": "12px", "padding": "16px"},
                         children=[html.P("Run an audit to see results.",
                                          style={"color": COLORS["text_dim"], "fontSize": "13px", "margin": "0"})]),
            ]),
            # Main content
            html.Div(style={"flex": "1", "padding": "24px 28px", "overflowY": "auto"}, children=[
                dcc.Tabs(id="tabs", value="tab-ensemble", style={"marginBottom": "16px"},
                         colors={"border": COLORS["card_border"], "primary": COLORS["accent"], "background": COLORS["card"]},
                         children=[dcc.Tab(label=lbl, value=val, style=TAB_STYLE, selected_style=TAB_SEL)
                                   for lbl, val in TAB_DEFS]),
                # Panels
                html.Div(id="panel-ensemble", children=[_make_graph("ensemble-graph")]),
                html.Div(id="panel-pulsar", style={"display": "none"}, children=[
                    dcc.Dropdown(id="pulsar-select", options=[], value=None, clearable=False,
                                 style={"backgroundColor": "#1e293b", "borderRadius": "8px",
                                        "marginBottom": "16px", "maxWidth": "400px"}),
                    _make_graph("pulsar-graph", "Select a pulsar.")]),
                html.Div(id="panel-sweep", style={"display": "none"}, children=[_make_graph("sweep-graph")]),
                html.Div(id="panel-skymap", style={"display": "none"}, children=[_make_graph("skymap-graph", "Run audit.")]),
                html.Div(id="panel-corr", style={"display": "none"}, children=[_make_graph("corr-graph", "Run audit.")]),
                html.Div(id="panel-null", style={"display": "none"}, children=[_make_graph("null-graph", "Run audit.")]),
                html.Div(id="panel-hd", style={"display": "none"}, children=[_make_graph("hd-graph", "Run audit.")]),
                html.Div(id="panel-table", style={"display": "none"}, children=[
                    dash_table.DataTable(
                        id="results-table", data=[], page_size=15,
                        columns=[{"name": c, "id": c} for c in
                                 ["Pulsar", "σ", "p-value", "log₁₀(BF)", "Bayes Verdict",
                                  "Window RMS", "Baseline RMS", "Artifact"]],
                        style_table={"overflowX": "auto", "borderRadius": "12px"},
                        style_header={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                                      "fontWeight": "600", "fontSize": "12px", "textTransform": "uppercase",
                                      "letterSpacing": "0.5px", "border": f"1px solid {COLORS['card_border']}",
                                      "padding": "10px 14px"},
                        style_data={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                    "fontSize": "13px", "fontFamily": "'JetBrains Mono', monospace",
                                    "border": f"1px solid {COLORS['card_border']}", "padding": "8px 14px"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{Artifact} contains "YES"'}, "color": COLORS["red"], "fontWeight": "bold"},
                            {"if": {"filter_query": '{Artifact} contains "No"'}, "color": COLORS["green"]},
                        ],
                        style_cell={"textAlign": "center"}),
                    html.Div(id="verdict-banner", style={"marginTop": "16px"})]),
            ]),
        ]),
        dcc.Store(id="audit-config", data=None),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(*[Output(pid, "style") for pid in PANEL_IDS], Input("tabs", "value"))
def switch_tab(tab):
    s, h = {"display": "block"}, {"display": "none"}
    tab_to_idx = {t[1]: i for i, t in enumerate(TAB_DEFS)}
    return tuple(s if i == tab_to_idx.get(tab, 0) else h for i in range(len(TAB_DEFS)))


@callback(Output("audit-config", "data"), Output("loading-text", "children"),
          Input("run-button", "n_clicks"),
          State("event-dropdown", "value"), State("window-dropdown", "value"),
          State("simulate-lag-check", "value"), prevent_initial_call=True)
def run_audit_cb(n_clicks, event, window, sim_check):
    if n_clicks == 0:
        return no_update, ""
    gw_time = KNOWN_EVENTS[event]
    sim = "sim" in (sim_check or [])
    ensemble = _load_ensemble(gw_time, simulate_lag=sim)
    if not ensemble:
        return no_update, "❌ No pulsar data."
    _load_strain(gw_time)

    from demiurge_trace import auditor
    results = auditor.audit_ensemble(gw_time, ensemble, window_seconds=window,
                                     monte_carlo=True, mc_iterations=10000)

    # Sliding window sweep
    sweeps = {}
    for psr, (t, r) in ensemble.items():
        sweeps[psr] = auditor.sliding_window_sweep(gw_time, t, r, window, 200)
    _SWEEP_CACHE[(gw_time, window)] = sweeps

    # Correlation matrix
    _CORR_CACHE[(gw_time, window)] = auditor.correlation_matrix(ensemble, gw_time, window)

    # Null distribution (ensemble mean sigma at 100 random times)
    null_sigmas = []
    import numpy as np
    rng = np.random.default_rng(42)
    all_t = np.concatenate([t for t, _ in ensemble.values()])
    t_lo, t_hi = float(all_t.min()) + window, float(all_t.max()) - window
    for rand_t in rng.uniform(t_lo, t_hi, 100):
        trial = auditor.audit_ensemble(rand_t, ensemble, window, monte_carlo=False)
        null_sigmas.append(trial["ensemble_sigma"])
    _NULL_CACHE[(gw_time, window)] = {"null_sigmas": np.array(null_sigmas),
                                       "mean": float(np.mean(null_sigmas)),
                                       "std": float(np.std(null_sigmas))}

    # Hellings-Downs
    _HD_CACHE[(gw_time, window)] = auditor.hellings_downs_analysis(
        ensemble, PULSAR_COORDS, gw_time, window)

    config_key = f"{event}_{window}_{sim}"
    _AUDIT_RESULTS[config_key] = results

    config = {
        "event": event, "window": window, "sim": sim, "key": config_key,
        "ensemble_sigma": results["ensemble_sigma"],
        "n_pulsars_in_window": results["n_pulsars_in_window"],
        "n_detectors_above_1s": results["n_detectors_above_1s"],
        "is_ensemble_artifact": results["is_ensemble_artifact"],
        "pulsar_names": list(ensemble.keys()),
        "pulsar_summaries": {
            psr: {k: results["pulsar_results"].get(psr, {}).get(k, 0)
                  for k in ["sigma", "window_rms", "baseline_rms", "is_artifact",
                            "p_value", "bayes_factor", "log_bf", "bf_interpretation"]}
            for psr in ensemble.keys()
        },
    }
    return config, f"✅ {len(ensemble)} pulsars | MC + sweep + corr + null + HD done."


@callback(Output("status-badge", "children"), Output("status-badge", "style"), Input("audit-config", "data"))
def update_badge(cfg):
    base = {"padding": "8px 20px", "borderRadius": "20px", "fontSize": "13px", "fontWeight": "600", "letterSpacing": "0.5px"}
    if cfg is None:
        return "AWAITING AUDIT", {**base, "backgroundColor": "rgba(148,163,184,0.15)", "color": COLORS["text_dim"]}
    if cfg["is_ensemble_artifact"]:
        return "🚨 ARTIFACT", {**base, "backgroundColor": "rgba(239,68,68,0.2)", "color": COLORS["red"], "border": f"1px solid {COLORS['red']}"}
    return "✓ NULL", {**base, "backgroundColor": "rgba(16,185,129,0.15)", "color": COLORS["green"], "border": f"1px solid {COLORS['green']}"}


@callback(Output("results-card", "children"), Input("audit-config", "data"))
def update_card(cfg):
    if cfg is None:
        return html.P("Run an audit.", style={"color": COLORS["text_dim"], "fontSize": "13px", "margin": "0"})
    sc = COLORS["green"] if abs(cfg["ensemble_sigma"]) < 1 else (COLORS["yellow"] if abs(cfg["ensemble_sigma"]) < 3 else COLORS["red"])
    return [
        html.H3("Results", style={"margin": "0 0 12px 0", "fontSize": "12px", "fontWeight": "600",
                 "textTransform": "uppercase", "letterSpacing": "1px", "color": COLORS["text_dim"]}),
        _result_row("Event", cfg["event"]),
        _result_row("Window", f"{cfg['window'] / 86400:.1f}d"),
        _result_row("Pulsars", str(cfg["n_pulsars_in_window"])),
        _result_row("Ensemble σ", f"{cfg['ensemble_sigma']:.4f}", sc),
        _result_row("> 1σ", str(cfg["n_detectors_above_1s"])),
        html.Hr(style={"borderColor": COLORS["card_border"], "margin": "10px 0"}),
        _result_row("Verdict", "ARTIFACT" if cfg["is_ensemble_artifact"] else "NULL",
                     COLORS["red"] if cfg["is_ensemble_artifact"] else COLORS["green"]),
    ]


@callback(Output("ensemble-graph", "figure"), Input("audit-config", "data"))
def update_ensemble(cfg):
    if cfg is None: return empty_fig()
    gw_time = KNOWN_EVENTS[cfg["event"]]
    return build_ensemble_fig(_ENSEMBLE_CACHE.get((gw_time, False), {}),
                              _STRAIN_CACHE.get(gw_time), gw_time, cfg["window"])


@callback(Output("pulsar-select", "options"), Output("pulsar-select", "value"), Input("audit-config", "data"))
def update_psr_dd(cfg):
    if cfg is None: return [], None
    names = cfg["pulsar_names"]
    return [{"label": p, "value": p} for p in names], names[0] if names else None


@callback(Output("pulsar-graph", "figure"), Input("pulsar-select", "value"), Input("audit-config", "data"))
def update_psr(psr, cfg):
    if cfg is None or psr is None: return empty_fig("Select a pulsar.")
    gw_time = KNOWN_EVENTS[cfg["event"]]
    return build_pulsar_fig(_ENSEMBLE_CACHE.get((gw_time, False), {}), gw_time, psr, cfg["window"],
                            _AUDIT_RESULTS.get(cfg["key"], {}))


@callback(Output("sweep-graph", "figure"), Input("audit-config", "data"))
def update_sweep(cfg):
    if cfg is None: return empty_fig()
    return build_sweep_fig(_SWEEP_CACHE.get((KNOWN_EVENTS[cfg["event"]], cfg["window"])))


@callback(Output("skymap-graph", "figure"), Input("audit-config", "data"))
def update_skymap(cfg):
    if cfg is None: return empty_fig("Run audit.")
    gw_time = KNOWN_EVENTS[cfg["event"]]
    return build_sky_map(_ENSEMBLE_CACHE.get((gw_time, False), {}), _AUDIT_RESULTS.get(cfg["key"], {}))


@callback(Output("corr-graph", "figure"), Input("audit-config", "data"))
def update_corr(cfg):
    if cfg is None: return empty_fig("Run audit.")
    return build_correlation_fig(_CORR_CACHE.get((KNOWN_EVENTS[cfg["event"]], cfg["window"])))


@callback(Output("null-graph", "figure"), Input("audit-config", "data"))
def update_null(cfg):
    if cfg is None: return empty_fig("Run audit.")
    null = _NULL_CACHE.get((KNOWN_EVENTS[cfg["event"]], cfg["window"]))
    return build_null_dist_fig(null, cfg["ensemble_sigma"])


@callback(Output("hd-graph", "figure"), Input("audit-config", "data"))
def update_hd(cfg):
    if cfg is None: return empty_fig("Run audit.")
    return build_hellings_downs_fig(_HD_CACHE.get((KNOWN_EVENTS[cfg["event"]], cfg["window"])))


@callback(Output("results-table", "data"), Output("verdict-banner", "children"), Input("audit-config", "data"))
def update_table(cfg):
    if cfg is None:
        return [], html.P("Run an audit.", style={"color": COLORS["text_dim"]})
    rows = []
    for psr, s in cfg["pulsar_summaries"].items():
        p_val = s.get("p_value")
        rows.append({
            "Pulsar": psr, "σ": f"{s['sigma']:.4f}",
            "p-value": f"{p_val:.4f}" if p_val is not None else "—",
            "log₁₀(BF)": f"{s.get('log_bf', 0):.3f}",
            "Bayes Verdict": s.get("bf_interpretation", "—"),
            "Window RMS": f"{s['window_rms']:.2e}",
            "Baseline RMS": f"{s['baseline_rms']:.2e}",
            "Artifact": "🚨 YES" if s.get("is_artifact") else "✓ No",
        })
    banner = html.Div(
        style={"padding": "16px", "backgroundColor": "rgba(99,102,241,0.08)", "borderRadius": "12px",
               "border": f"1px solid {COLORS['card_border']}"},
        children=[
            html.H3(f"{'COHERENT ARTIFACT DETECTED' if cfg['is_ensemble_artifact'] else 'Universal Clock Stability Confirmed'}",
                     style={"color": COLORS["red"] if cfg["is_ensemble_artifact"] else COLORS["green"],
                            "margin": "0 0 6px 0", "fontSize": "16px"}),
            html.P(f"σ̄ = {cfg['ensemble_sigma']:.4f}  |  {cfg['n_pulsars_in_window']} pulsars  |  "
                   f"{cfg['n_detectors_above_1s']} > 1σ  |  MC: 10k  |  Sim: {'ON' if cfg['sim'] else 'OFF'}",
                   style={"color": COLORS["text_dim"], "margin": "0", "fontSize": "13px",
                          "fontFamily": "'JetBrains Mono', monospace"})])
    return rows, banner


# Export callbacks
@callback(Output("download-data", "data"), Input("btn-csv", "n_clicks"), Input("btn-json", "n_clicks"),
          State("audit-config", "data"), prevent_initial_call=True)
def export_data(csv_clicks, json_clicks, cfg):
    from dash import ctx
    if cfg is None: return no_update
    summaries = cfg.get("pulsar_summaries", {})
    if ctx.triggered_id == "btn-csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Pulsar", "sigma", "p_value", "log10_bf", "bayes_verdict",
                          "window_rms", "baseline_rms", "artifact"])
        for psr, s in summaries.items():
            writer.writerow([psr, f"{s['sigma']:.4f}", f"{s.get('p_value', ''):.4f}" if s.get('p_value') else "",
                              f"{s.get('log_bf', 0):.3f}", s.get("bf_interpretation", ""),
                              f"{s['window_rms']:.2e}", f"{s['baseline_rms']:.2e}",
                              "YES" if s.get("is_artifact") else "No"])
        return dict(content=output.getvalue(), filename=f"demiurge_{cfg['event']}.csv")
    elif ctx.triggered_id == "btn-json":
        export = {"event": cfg["event"], "window_seconds": cfg["window"],
                  "ensemble_sigma": cfg["ensemble_sigma"],
                  "is_artifact": cfg["is_ensemble_artifact"], "pulsars": summaries}
        return dict(content=json.dumps(export, indent=2), filename=f"demiurge_{cfg['event']}.json")
    return no_update


# ---------------------------------------------------------------------------
# Pre-load and start
# ---------------------------------------------------------------------------
def _preload():
    t0 = time.time()
    for name, gps in KNOWN_EVENTS.items():
        print(f"  Loading ensemble for {name}...")
        try: _load_ensemble(gps, simulate_lag=False)
        except Exception as e: print(f"    ⚠ {e}")
        print(f"  Loading strain for {name}...")
        try: _load_strain(gps)
        except Exception as e: print(f"    ⚠ Strain unavailable: {e}")
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
