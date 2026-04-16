"""Coordinate mapping helpers for LILA BLACK minimaps."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

MAP_BOUNDS: Dict[str, Dict[str, float]] = {
    "AmbroseValley": {
        "x_min": -321.835,
        "x_max": 272.42,
        "z_min": -375.02,
        "z_max": 337.854,
    },
    "GrandRift": {
        "x_min": -225.89,
        "x_max": 252.64,
        "z_min": -183.90,
        "z_max": 169.5017,
    },
    "Lockdown": {
        "x_min": -406.63,
        "x_max": 328.013,
        "z_min": -255.96,
        "z_max": 329.047,
    },
}


def map_points_to_pixels(
    x_values: pd.Series,
    z_values: pd.Series,
    map_id: str,
    image_width: int,
    image_height: int,
) -> Tuple[pd.Series, pd.Series]:
    """Map game coordinates to image pixels using X/Z bounds and Z-flip."""
    if map_id not in MAP_BOUNDS:
        raise ValueError(f"Unknown map_id: {map_id}")

    bounds = MAP_BOUNDS[map_id]

    pixel_x = (
        (x_values - bounds["x_min"]) / (bounds["x_max"] - bounds["x_min"]) * image_width
    )

    pixel_z = (
        1 - (z_values - bounds["z_min"]) / (bounds["z_max"] - bounds["z_min"])
    ) * image_height

    return pixel_x, pixel_z
