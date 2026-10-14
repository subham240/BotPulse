"""
Interactive Cybersecurity Visualizations for BotPulse.

Provides styled Plotly visualizations following the BotPulse Dark Cyber Design System:
- Deep navy background (#0b1120, #0f172a)
- Indigo (#6366f1) and Cyan (#06b6d4) accent colors
- Emerald (#10b981) for verified humans, Rose (#f43f5e) for bot threats
"""

from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.feature_mapping import CLASS_NAMES


CYBER_THEME = {
    "bg_dark": "#0b1120",
    "card_bg": "#111827",
    "border": "#1e293b",
    "text_main": "#f8fafc",
    "text_muted": "#94a3b8",
    "cyan": "#06b6d4",
    "cyan_glow": "#22d3ee",
    "indigo": "#6366f1",
    "purple": "#a855f7",
    "emerald": "#10b981",
    "rose": "#f43f5e",
    "amber": "#f59e0b",
}


def apply_cyber_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Applies common dark cybersecurity layout formatting to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": CYBER_THEME["text_main"], "family": "Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor=CYBER_THEME["card_bg"],
        plot_bgcolor="rgba(11, 17, 32, 0.7)",
        font={"color": CYBER_THEME["text_muted"], "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        xaxis=dict(
            gridcolor="rgba(30, 41, 59, 0.6)",
            zerolinecolor=CYBER_THEME["border"],
            color=CYBER_THEME["text_muted"],
        ),
        yaxis=dict(
            gridcolor="rgba(30, 41, 59, 0.6)",
            zerolinecolor=CYBER_THEME["border"],
            color=CYBER_THEME["text_muted"],
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor=CYBER_THEME["border"],
            borderwidth=1,
            font={"color": CYBER_THEME["text_main"]},
        ),
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], title: str = "Test Set Confusion Matrix") -> go.Figure:
    """Renders a styled interactive confusion matrix heatmap."""
    labels = ["Human", "Web Bot"]
    z = np.array(cm)

    # Annotations with percentages
    total = np.sum(z)
    annotations = []
    for i in range(2):
        for j in range(2):
            count = z[i][j]
            pct = (count / total) * 100.0
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{count:,}</b><br><span style='font-size:11px'>({pct:.1f}%)</span>",
                    showarrow=False,
                    font=dict(color="#ffffff", size=14),
                )
            )

    colorscale = [
        [0.0, "#0b192c"],
        [0.5, "#1e3a8a"],
        [1.0, "#06b6d4"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=colorscale,
            showscale=False,
            hoverongaps=False,
        )
    )
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(title="Predicted Label", side="bottom")
    fig.update_yaxes(title="Actual Ground Truth", autorange="reversed")
    return apply_cyber_layout(fig, title=title, height=360)


def plot_feature_importances(
    importances: Dict[str, Dict[str, float]],
    top_n: int = 12,
    title: str = "Top Predictive Behavioral Features"
) -> go.Figure:
    """Renders a horizontal bar chart comparing feature importance across algorithms."""
    items = list(importances.items())[:top_n]
    items.reverse()  # Top at top of bar chart

    features = [k for k, _ in items]
    rf_vals = [v.get("random_forest", 0.0) for _, v in items]
    xgb_vals = [v.get("xgboost", 0.0) for _, v in items]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=features,
            x=rf_vals,
            name="Random Forest",
            orientation="h",
            marker=dict(color=CYBER_THEME["indigo"], line=dict(color="#4338ca", width=1)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=features,
            x=xgb_vals,
            name="XGBoost",
            orientation="h",
            marker=dict(color=CYBER_THEME["cyan"], line=dict(color="#0891b2", width=1)),
        )
    )

    fig.update_layout(barmode="group", xaxis_title="Gini / Gain Importance")
    return apply_cyber_layout(fig, title=title, height=480)


def plot_distribution_donut(
    counts: Dict[str, int],
    title: str = "Distribution",
    color_palette: Optional[List[str]] = None
) -> go.Figure:
    """Renders an interactive glassmorphic donut chart."""
    labels = list(counts.keys())
    values = list(counts.values())

    if color_palette is None:
        color_palette = [CYBER_THEME["cyan"], CYBER_THEME["indigo"], CYBER_THEME["emerald"], CYBER_THEME["rose"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=color_palette[:len(labels)], line=dict(color=CYBER_THEME["bg_dark"], width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(color=CYBER_THEME["text_main"], size=12),
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        annotations=[
            dict(
                text=f"<b>{sum(values):,}</b><br><span style='font-size:10px'>Total</span>",
                x=0.5,
                y=0.5,
                font_size=15,
                showarrow=False,
                font=dict(color=CYBER_THEME["text_main"]),
            )
        ],
    )
    return apply_cyber_layout(fig, title=title, height=360)


def plot_algorithm_comparison(metrics_rf: Dict[str, Any], metrics_xgb: Dict[str, Any]) -> go.Figure:
    """Renders grouped bar chart comparing performance metrics between models."""
    categories = ["Accuracy", "Macro-F1", "Precision (Macro)", "Recall (Macro)", "Weighted-F1"]

    rf_scores = [
        metrics_rf["accuracy"],
        metrics_rf["macro_f1"],
        metrics_rf["precision_macro"],
        metrics_rf["recall_macro"],
        metrics_rf["weighted_f1"],
    ]
    xgb_scores = [
        metrics_xgb["accuracy"],
        metrics_xgb["macro_f1"],
        metrics_xgb["precision_macro"],
        metrics_xgb["recall_macro"],
        metrics_xgb["weighted_f1"],
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Random Forest",
                x=categories,
                y=rf_scores,
                marker_color=CYBER_THEME["indigo"],
                text=[f"{s:.3f}" for s in rf_scores],
                textposition="auto",
            ),
            go.Bar(
                name="XGBoost",
                x=categories,
                y=xgb_scores,
                marker_color=CYBER_THEME["cyan"],
                text=[f"{s:.3f}" for s in xgb_scores],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(barmode="group", yaxis_range=[0.75, 1.02], yaxis_title="Score")
    return apply_cyber_layout(fig, title="Held-Out Model Performance Benchmark", height=400)


def plot_confidence_gauge(prob_robot: float, title: str = "Bot Threat Probability") -> go.Figure:
    """Renders a cybersecurity threat meter / radial gauge."""
    pct = prob_robot * 100.0

    if prob_robot < 0.35:
        bar_color = CYBER_THEME["emerald"]
        verdict = "Low Risk (Likely Human)"
    elif prob_robot < 0.65:
        bar_color = CYBER_THEME["amber"]
        verdict = "Suspicious Traffic"
    else:
        bar_color = CYBER_THEME["rose"]
        verdict = "High Risk (Automated Bot)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"color": CYBER_THEME["text_main"], "size": 36}},
            title={"text": f"<b>{verdict}</b>", "font": {"size": 15, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": CYBER_THEME["text_muted"]},
                "bar": {"color": bar_color},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": CYBER_THEME["border"],
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [65, 100], "color": "rgba(244, 63, 94, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": CYBER_THEME["cyan"], "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    return apply_cyber_layout(fig, title=title, height=290)


def plot_decision_factors_waterfall(factors: List[Dict[str, Any]], title: str = "Top Decision Driving Factors") -> go.Figure:
    """Renders deviation comparison of top influential features relative to human baseline."""
    labels = [f["label"] for f in factors]
    diffs = [f["relative_difference_pct"] for f in factors]
    colors = [CYBER_THEME["rose"] if d > 0 else CYBER_THEME["cyan"] for d in diffs]

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=diffs,
                orientation="h",
                marker_color=colors,
                text=[f"{d:+.1f}%" for d in diffs],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Deviation from Empirical Human Baseline (%)",
        yaxis_autorange="reversed",
    )
    return apply_cyber_layout(fig, title=title, height=360)


def plot_threat_histogram(confidences: List[float], title: str = "External Traffic Threat Score Distribution") -> go.Figure:
    """Renders a histogram of bot threat scores across an uploaded external dataset."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=[c * 100.0 for c in confidences],
                nbinsx=15,
                marker=dict(
                    color=CYBER_THEME["cyan"],
                    line=dict(color="#0891b2", width=1.5),
                ),
                opacity=0.85,
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Predicted Bot Threat Score (%)",
        yaxis_title="Number of Sessions",
        bargap=0.08,
    )
    return apply_cyber_layout(fig, title=title, height=340)


def plot_traffic_breakdown_bar(summary_stats: Dict[str, Dict[str, float]], title: str = "Behavioral Comparison: Humans vs Bots") -> go.Figure:
    """Compares average feature metrics of detected humans vs detected bots in external dataset."""
    categories = list(summary_stats.keys())
    human_vals = [summary_stats[c].get("human", 0.0) for c in categories]
    bot_vals = [summary_stats[c].get("bot", 0.0) for c in categories]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Detected Humans",
                x=categories,
                y=human_vals,
                marker_color=CYBER_THEME["emerald"],
            ),
            go.Bar(
                name="Detected Bots",
                x=categories,
                y=bot_vals,
                marker_color=CYBER_THEME["rose"],
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Normalized Value",
    )
    return apply_cyber_layout(fig, title=title, height=350)


def plot_mouse_trajectories_side_by_side(
    h_xs: List[int],
    h_ys: List[int],
    b_xs: List[int],
    b_ys: List[int],
    title: str = "Mouse Movement Comparison: Genuine Human vs Advanced Bot"
) -> go.Figure:
    """Renders side-by-side 2D browser canvas trajectories comparing Human and Advanced Bot."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("🟢 Genuine Human (Organic Curves & Tremors)", "🔴 Advanced Bot (Synthetic Mathematical Lines)"),
        horizontal_spacing=0.08,
    )

    # Human trajectory
    if h_xs and h_ys:
        fig.add_trace(
            go.Scatter(
                x=h_xs,
                y=h_ys,
                mode="lines+markers",
                name="Human Path",
                line=dict(color=CYBER_THEME["emerald"], width=2.5, shape="spline"),
                marker=dict(size=4, color="#34d399"),
            ),
            row=1,
            col=1,
        )
        # Highlight start and end
        fig.add_trace(
            go.Scatter(
                x=[h_xs[0], h_xs[-1]],
                y=[h_ys[0], h_ys[-1]],
                mode="markers+text",
                name="Human Start/End",
                text=["START", "END"],
                textposition="top center",
                marker=dict(size=10, color=["#10b981", "#38bdf8"], symbol="diamond"),
            ),
            row=1,
            col=1,
        )

    # Advanced Bot trajectory
    if b_xs and b_ys:
        fig.add_trace(
            go.Scatter(
                x=b_xs,
                y=b_ys,
                mode="lines+markers",
                name="Advanced Bot Path",
                line=dict(color=CYBER_THEME["rose"], width=2.5),
                marker=dict(size=4, color="#fb7185"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[b_xs[0], b_xs[-1]],
                y=[b_ys[0], b_ys[-1]],
                mode="markers+text",
                name="Bot Start/End",
                text=["START", "END"],
                textposition="top center",
                marker=dict(size=10, color=["#f43f5e", "#fbbf24"], symbol="diamond"),
            ),
            row=1,
            col=2,
        )

    # Invert Y axes because browser screen coordinates have (0,0) at the top-left
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(30, 41, 59, 0.6)", row=1, col=1, title="Browser Y (pixels)")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(30, 41, 59, 0.6)", row=1, col=2, title="Browser Y (pixels)")
    fig.update_xaxes(gridcolor="rgba(30, 41, 59, 0.6)", row=1, col=1, title="Browser X (pixels)")
    fig.update_xaxes(gridcolor="rgba(30, 41, 59, 0.6)", row=1, col=2, title="Browser X (pixels)")

    fig.update_layout(
        paper_bgcolor=CYBER_THEME["card_bg"],
        plot_bgcolor="rgba(11, 17, 32, 0.8)",
        font={"color": CYBER_THEME["text_muted"], "family": "Inter, sans-serif"},
        height=480,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        showlegend=False,
    )
    return fig


