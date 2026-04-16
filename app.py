"""LILA Games Player Journey Visualization Tool."""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_plotly_events import plotly_events

from utils.coordinate_mapper import map_points_to_pixels
from utils.data_loader import (
    build_index,
    build_match_display_map,
    default_data_dir,
    get_available_dates,
    get_matches,
    load_filtered_match_data,
)
from utils.heatmap import build_heatmap_trace

st.set_page_config(
    page_title="LILA BLACK - Player Journey Visualization",
    page_icon=":video_game:",
    layout="wide",
)

MAP_OPTIONS = ["AmbroseValley", "GrandRift", "Lockdown"]

EVENT_STYLE = {
    "BotKill": {"color": "#ff9f1c", "symbol": "triangle-up", "size": 11},
    "BotKilled": {"color": "#ff7f11", "symbol": "triangle-down", "size": 11},
    "Kill": {"color": "#ff2b2b", "symbol": "circle", "size": 10},
    "Killed": {"color": "#8b0000", "symbol": "circle", "size": 9},
    "KilledByStorm": {"color": "#9b5de5", "symbol": "diamond", "size": 10},
    "Loot": {"color": "#ffd60a", "symbol": "star", "size": 10},
}

HEATMAP_OPTIONS = ["Kill heatmap", "Death heatmap", "Traffic heatmap", "Loot heatmap"]


@st.cache_data(show_spinner=False)
def load_minimap_image(project_root: str, map_id: str) -> np.ndarray:
    """Load minimap image for selected map as numpy array."""
    map_path = os.path.join(project_root, "maps", f"{map_id}.png")
    image = Image.open(map_path).convert("RGB")
    return np.array(image)


def short_user(user_id: str) -> str:
    if not isinstance(user_id, str):
        return "unknown"
    return user_id[:8]


def color_map_for_players(user_ids: List[str]) -> Dict[str, str]:
    palette = (
        [
            "#00e5ff",
            "#ff6b6b",
            "#ffe66d",
            "#4ecdc4",
            "#ff9f1c",
            "#a29bfe",
            "#7bed9f",
            "#feca57",
            "#48dbfb",
            "#ff7f50",
            "#54a0ff",
            "#eccc68",
        ]
        * 50
    )
    return {user_id: palette[index] for index, user_id in enumerate(sorted(user_ids))}


def filter_by_player_type(dataframe: pd.DataFrame, player_type: str) -> pd.DataFrame:
    if player_type == "Humans only":
        return dataframe[dataframe["is_bot"] == False]
    if player_type == "Bots only":
        return dataframe[dataframe["is_bot"] == True]
    return dataframe


def build_stats(dataframe: pd.DataFrame) -> Dict[str, str]:
    if dataframe.empty:
        return {
            "players": "0 (Humans: 0 | Bots: 0)",
            "kills": "0 (Human kills: 0 | Bot kills: 0)",
            "loot": "0",
            "storm": "0",
            "duration": "0s",
        }

    player_flags = dataframe[["user_id", "is_bot"]].drop_duplicates()
    human_players = int((player_flags["is_bot"] == False).sum())
    bot_players = int((player_flags["is_bot"] == True).sum())

    kill_rows = dataframe[dataframe["event"].isin(["Kill", "BotKill"])]
    human_kills = int((kill_rows["is_bot"] == False).sum())
    bot_kills = int((kill_rows["is_bot"] == True).sum())

    loot_count = int((dataframe["event"] == "Loot").sum())
    storm_deaths = int((dataframe["event"] == "KilledByStorm").sum())

    ts_values = dataframe["ts"].dropna()
    if ts_values.empty:
        duration_text = "0s"
    else:
        seconds = int((ts_values.max() - ts_values.min()).total_seconds())
        minutes, rem_seconds = divmod(max(seconds, 0), 60)
        duration_text = f"{minutes}m {rem_seconds}s"

    return {
        "players": f"{human_players + bot_players} (Humans: {human_players} | Bots: {bot_players})",
        "kills": f"{len(kill_rows)} (Human kills: {human_kills} | Bot kills: {bot_kills})",
        "loot": str(loot_count),
        "storm": str(storm_deaths),
        "duration": duration_text,
    }


def add_minimap_background(fig: go.Figure, image: np.ndarray, width: int, height: int) -> None:
    """Attach minimap image as figure background."""
    fig.add_layout_image(
        dict(
            source=image,
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            xref="x",
            yref="y",
            xanchor="left",
            yanchor="bottom",
            layer="below",
            sizing="stretch",
            opacity=1.0,
        )
    )

    fig.update_xaxes(range=[0, width], visible=False)
    fig.update_yaxes(range=[height, 0], visible=False, scaleanchor="x", scaleratio=1)


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))

    st.title("LILA BLACK - Player Journey Visualization")
    st.caption("Interactive map analytics for LILA Games level design teams")

    with st.sidebar:
        st.header("Filters")

        map_id = st.selectbox("Map", MAP_OPTIONS, index=0)

        with st.expander("Data Source", expanded=False):
            default_dir = default_data_dir(project_root)
            data_dir = st.text_input("Data folder", value=default_dir)
            st.caption("Tip: Use a folder containing date-wise parquet files.")

        index_df = build_index(data_dir)
        available_dates = get_available_dates(index_df, map_id)

        if not available_dates:
            st.warning("No dates found for this map in the selected data folder.")
            st.stop()

        date_value = st.selectbox("Date", available_dates, index=0)

        match_ids = get_matches(index_df, map_id, date_value)
        if not match_ids:
            st.warning("No matches found for this map/date.")
            st.stop()

        display_map = build_match_display_map(match_ids)
        display_keys = sorted(display_map.keys())
        selected_display = st.selectbox("Match", display_keys, index=0)
        selected_match_id = display_map[selected_display]

        player_type = st.radio("Player type", ["All", "Humans only", "Bots only"], index=0)

        selected_match_df = load_filtered_match_data(data_dir, map_id, date_value, selected_match_id)

        if selected_match_df.empty:
            st.warning("No records found for this exact filter combination.")
            st.stop()

        all_events = sorted(selected_match_df["event"].dropna().unique().tolist())
        default_events = all_events.copy()

        selected_events = st.multiselect(
            "Event types",
            options=all_events,
            default=default_events,
        )

        view_mode = st.radio("View mode", ["Paths + Events", "Heatmap"], index=0)

        heatmap_type = None
        if view_mode == "Heatmap":
            heatmap_type = st.radio("Heatmap type", HEATMAP_OPTIONS, index=0)

    filtered_df = filter_by_player_type(selected_match_df, player_type)

    if selected_events:
        filtered_df = filtered_df[filtered_df["event"].isin(selected_events)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    # Timeline controls operate on the full selected match (before event filtering)
    timeline_source = filter_by_player_type(selected_match_df, player_type)
    ts_values = sorted([ts for ts in timeline_source["ts"].dropna().unique()])

    if "timeline_idx" not in st.session_state:
        st.session_state["timeline_idx"] = 0
    if "playing" not in st.session_state:
        st.session_state["playing"] = False
    if "play_speed" not in st.session_state:
        st.session_state["play_speed"] = 1.0
    if "highlight_user" not in st.session_state:
        st.session_state["highlight_user"] = None

    if ts_values:
        max_idx = len(ts_values) - 1
        st.session_state["timeline_idx"] = min(st.session_state["timeline_idx"], max_idx)

        controls_col1, controls_col2, controls_col3 = st.columns([2, 1, 1])

        with controls_col1:
            timeline_idx = st.slider(
                "Match timeline",
                min_value=0,
                max_value=max_idx,
                value=st.session_state["timeline_idx"],
                key="timeline_slider",
            )
            st.session_state["timeline_idx"] = timeline_idx

        with controls_col2:
            play_pause_label = "Pause" if st.session_state["playing"] else "Play"
            if st.button(play_pause_label, use_container_width=True):
                st.session_state["playing"] = not st.session_state["playing"]

        with controls_col3:
            speed = st.selectbox("Speed", [0.5, 1.0, 2.0], index=[0.5, 1.0, 2.0].index(st.session_state["play_speed"]))
            st.session_state["play_speed"] = float(speed)

        current_ts = ts_values[st.session_state["timeline_idx"]]
        base_ts = ts_values[0]
        elapsed_seconds = max((current_ts - base_ts).total_seconds(), 0)
        st.caption(f"Showing events up to +{elapsed_seconds:.2f}s")

        visible_df = filtered_df[filtered_df["ts"] <= current_ts].copy()
    else:
        visible_df = filtered_df.copy()
        st.caption("No valid timestamps available; showing complete filtered match.")

    stats = build_stats(selected_match_df)
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    stat_col1.metric("Total players", stats["players"])
    stat_col2.metric("Total kills", stats["kills"])
    stat_col3.metric("Loot pickups", stats["loot"])
    stat_col4.metric("Storm deaths", stats["storm"])
    stat_col5.metric("Match duration", stats["duration"])

    map_image = load_minimap_image(project_root, map_id)
    image_height, image_width = map_image.shape[0], map_image.shape[1]

    fig = go.Figure()
    add_minimap_background(fig, map_image, image_width, image_height)

    if view_mode == "Heatmap":
        heatmap_trace = build_heatmap_trace(
            visible_df,
            map_id,
            image_width,
            image_height,
            heatmap_type or "Traffic heatmap",
        )
        if heatmap_trace is not None:
            fig.add_trace(heatmap_trace)
        else:
            st.info("No events available for the selected heatmap type and filters.")
    else:
        # Human paths use Position events, bot paths use BotPosition and fallback bot Position rows.
        path_rows = visible_df[
            ((visible_df["event"] == "Position") & (visible_df["is_bot"] == False))
            | ((visible_df["event"].isin(["BotPosition", "Position"])) & (visible_df["is_bot"] == True))
        ].copy()

        event_rows = visible_df[visible_df["event"].isin(EVENT_STYLE.keys())].copy()

        if not path_rows.empty:
            path_rows["pixel_x"], path_rows["pixel_z"] = map_points_to_pixels(
                path_rows["x"], path_rows["z"], map_id, image_width, image_height
            )

            players = path_rows["user_id"].dropna().unique().tolist()
            player_colors = color_map_for_players(players)
            highlighted_user = st.session_state.get("highlight_user")

            for user_id in players:
                user_path = path_rows[path_rows["user_id"] == user_id].sort_values("ts")
                if user_path.empty:
                    continue

                is_bot = bool(user_path["is_bot"].iloc[0])
                base_opacity = 0.35 if is_bot else 0.95
                dash_style = "dash" if is_bot else "solid"
                line_width = 1.6 if is_bot else 2.8

                if highlighted_user and highlighted_user != user_id:
                    base_opacity = 0.08
                if highlighted_user and highlighted_user == user_id:
                    base_opacity = 1.0
                    line_width = 4.0

                fig.add_trace(
                    go.Scattergl(
                        x=user_path["pixel_x"],
                        y=user_path["pixel_z"],
                        mode="lines",
                        line={
                            "color": player_colors[user_id],
                            "width": line_width,
                            "dash": dash_style,
                        },
                        opacity=base_opacity,
                        name=f"{short_user(user_id)} ({'BOT' if is_bot else 'HUM'})",
                        customdata=[[user_id]] * len(user_path),
                        hovertemplate=(
                            "User: %{text}<br>x=%{x:.1f}<br>z=%{y:.1f}<extra></extra>"
                        ),
                        text=[short_user(user_id)] * len(user_path),
                    )
                )

        if not event_rows.empty:
            event_rows["pixel_x"], event_rows["pixel_z"] = map_points_to_pixels(
                event_rows["x"], event_rows["z"], map_id, image_width, image_height
            )

            for event_name, style in EVENT_STYLE.items():
                subset = event_rows[event_rows["event"] == event_name]
                if subset.empty:
                    continue

                fig.add_trace(
                    go.Scattergl(
                        x=subset["pixel_x"],
                        y=subset["pixel_z"],
                        mode="markers",
                        marker={
                            "color": style["color"],
                            "symbol": style["symbol"],
                            "size": style["size"],
                            "line": {"width": 0.5, "color": "#111111"},
                        },
                        name=event_name,
                        hovertemplate=(
                            "Event: "
                            + event_name
                            + "<br>User: %{text}<br>x=%{x:.1f}<br>z=%{y:.1f}<extra></extra>"
                        ),
                        text=[short_user(user) for user in subset["user_id"].tolist()],
                    )
                )

    fig.update_layout(
        height=820,
        margin={"l": 0, "r": 0, "t": 10, "b": 10},
        legend={"orientation": "h", "y": -0.05},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    clicked_points = plotly_events(fig, click_event=True, select_event=False, hover_event=False)

    if clicked_points and view_mode == "Paths + Events":
        click_info = clicked_points[0]
        curve_number = click_info.get("curveNumber")
        point_number = click_info.get("pointNumber")

        if curve_number is not None and point_number is not None:
            trace = fig.data[curve_number]
            if hasattr(trace, "customdata") and trace.customdata is not None:
                selected = trace.customdata[point_number]
                if isinstance(selected, (list, tuple)) and selected:
                    st.session_state["highlight_user"] = selected[0]
                    st.rerun()

    if view_mode == "Paths + Events":
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("Clear highlighted player"):
                st.session_state["highlight_user"] = None
                st.rerun()
        with col_b:
            highlighted = st.session_state.get("highlight_user")
            if highlighted:
                st.caption(f"Highlighted player: {short_user(highlighted)}")
            else:
                st.caption("Tip: click a player path to highlight it.")

    # Playback runner: keeps app advancing while play mode is active.
    if ts_values and st.session_state.get("playing"):
        if st.session_state["timeline_idx"] < len(ts_values) - 1:
            interval = {0.5: 0.8, 1.0: 0.45, 2.0: 0.2}.get(st.session_state["play_speed"], 0.45)
            time.sleep(interval)
            st.session_state["timeline_idx"] += 1
            st.rerun()
        else:
            st.session_state["playing"] = False


if __name__ == "__main__":
    main()

