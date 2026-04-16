"""Heatmap trace generation for player journey visualization."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from utils.coordinate_mapper import map_points_to_pixels

HEATMAP_EVENT_MAP = {
    "Kill heatmap": ["Kill", "BotKill"],
    "Death heatmap": ["Killed", "BotKilled", "KilledByStorm"],
    "Traffic heatmap": ["Position", "BotPosition"],
    "Loot heatmap": ["Loot"],
}


def events_for_heatmap(dataframe: pd.DataFrame, heatmap_type: str) -> pd.DataFrame:
    """Return only rows relevant to the selected heatmap mode."""
    events = HEATMAP_EVENT_MAP.get(heatmap_type, [])
    if dataframe.empty or not events:
        return dataframe.iloc[0:0]
    return dataframe[dataframe["event"].isin(events)].copy()


def build_heatmap_trace(
    dataframe: pd.DataFrame,
    map_id: str,
    image_width: int,
    image_height: int,
    heatmap_type: str,
) -> Optional[go.Histogram2d]:
    """Build a Plotly 2D histogram trace overlay for the chosen heatmap."""
    filtered = events_for_heatmap(dataframe, heatmap_type)
    if filtered.empty:
        return None

    pixel_x, pixel_z = map_points_to_pixels(
        filtered["x"], filtered["z"], map_id, image_width, image_height
    )

    return go.Histogram2d(
        x=pixel_x,
        y=pixel_z,
        nbinsx=70,
        nbinsy=70,
        colorscale="Turbo",
        opacity=0.7,
        zsmooth="best",
        colorbar={"title": "Density"},
        hovertemplate="x=%{x:.1f}<br>z=%{y:.1f}<br>count=%{z}<extra></extra>",
    )
