# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mplsoccer import PyPizza
from scipy.stats import rankdata
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(layout="wide")
st.title("⚽ Scouting Dashboard")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\vicky\Desktop\Football-analytics\Datasets\SkillCorner Premier League 24-25 data\Player Metrics 24-25.csv")

df = load_data()

# =====================================================
# FEATURE GROUPS
# =====================================================
FEATURE_GROUPS = {
    "Finishing": [
        "shots_total_per90","shots_goals_per90","shots_conversion",
        "shots_self_created_pct","shots_transition_pct",
        "shots_difficulty_index","shots_pressure_index","shots_freedom_index"
    ],
    "Passing & Decision": [
        "passes_attempted_per90","avg_xpass_completion",
        "completion_minus_xpass_per_pass","total_pass_value_per90",
        "avg_decision_quality","chose_best_rate","option_selection_rate"
    ],
    "Involvement": [
        "times_targeted_per90"
    ],
    "Off-ball": [
        "offball_runs_total_per90","offball_distance_total_per90"
    ],
    "Possession Impact": [
        "possession_progression_total_per90",
        "possession_line_breaks_per90",
        "possession_threat_index"
    ],
    "Defensive": [
        "obe_total_events_per90",
        "obe_possession_losses_forced_per90",
        "obe_pressure_events_per90"
    ]
}

GROUP_COLORS = {
    "Finishing": "#E63946",
    "Passing & Decision": "#457B9D",
    "Involvement": "#2A9D8F",
    "Off-ball": "#F4A261",
    "Possession Impact": "#6A4C93",
    "Defensive": "#264653"
}

ALL_FEATURES = sum(FEATURE_GROUPS.values(), [])

# =====================================================
# HELPERS
# =====================================================
def compute_percentiles(df, features, position):
    df_pos = df[df["position_group"] == position].copy()

    for col in features:
        df_pos[col] = pd.to_numeric(df_pos[col], errors="coerce")
        df_pos[col] = df_pos[col].fillna(df_pos[col].median())
        df_pos[col] = rankdata(df_pos[col]) / len(df_pos) * 100

    return df_pos

def zscore(series):
    if series.std() == 0 or series.isna().all():
        return pd.Series([0]*len(series), index=series.index)
    return (series - series.mean()) / series.std()

def get_feature_colors(features):
    colors = []
    for f in features:
        for group, feats in FEATURE_GROUPS.items():
            if f in feats:
                colors.append(GROUP_COLORS[group])
                break
    return colors

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("Filters")

position = st.sidebar.selectbox(
    "Select Position",
    sorted(df["position_group"].dropna().unique())
)

players = df[df["position_group"] == position]["player_name"].sort_values().unique()

selected_players = st.sidebar.multiselect(
    "Select Players (2–3 recommended)",
    players,
    default=list(players[:2])
)

# =====================================================
# PIZZA PLOT
# =====================================================
def plot_pizza(df, players, position):

    df_pct = compute_percentiles(df, ALL_FEATURES, position)

    n = len(players)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 6), subplot_kw=dict(polar=True))

    if n == 1:
        axes = [axes]

    for ax, player in zip(axes, players):

        row = df_pct[df_pct["player_name"] == player]
        if row.empty:
            continue

        team = row["main_team"].values[0]

        values = row[ALL_FEATURES].values.flatten()
        values = np.nan_to_num(values, nan=0)

        labels = [f.replace("_","\n") for f in ALL_FEATURES]
        colors = get_feature_colors(ALL_FEATURES)

        baker = PyPizza(params=labels)

        baker.make_pizza(
            values,
            ax=ax,
            slice_colors=colors,
            value_colors=["black"]*len(values),
            value_bck_colors=colors
        )

        ax.set_title(f"{player} ({team})")

    st.pyplot(fig)

# =====================================================
# DISTRIBUTION PLOT
# =====================================================
def plot_distribution(df, players, position):

    df_pos = df[df["position_group"] == position].copy()

    for f in ALL_FEATURES:
        df_pos[f] = pd.to_numeric(df_pos[f], errors="coerce")
        df_pos[f+"_Z"] = zscore(df_pos[f])

    fig = go.Figure()

    for i, f in enumerate(ALL_FEATURES):

        fig.add_trace(go.Scatter(
            x=df_pos[f+"_Z"],
            y=[i]*len(df_pos),
            mode="markers",
            marker=dict(color="lightgray", size=6),
            showlegend=False
        ))

    colors = ["red","blue","green"]

    for j, player in enumerate(players):

        row = df_pos[df_pos["player_name"] == player]
        if row.empty:
            continue

        row = row.iloc[0]

        for i, f in enumerate(ALL_FEATURES):
            fig.add_trace(go.Scatter(
                x=[row[f+"_Z"]],
                y=[i],
                mode="markers",
                marker=dict(color=colors[j], size=12),
                name=player if i == 0 else None,
                showlegend=(i == 0)
            ))

    fig.update_layout(
        height=800,
        title=f"{position} Distribution",
        yaxis=dict(
            tickvals=list(range(len(ALL_FEATURES))),
            ticktext=[f.replace("_"," ") for f in ALL_FEATURES]
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# OUTPUT
# =====================================================
if selected_players:
    st.subheader("Pizza Profiles")
    plot_pizza(df, selected_players, position)

    st.subheader("Distribution Comparison")
    plot_distribution(df, selected_players, position)