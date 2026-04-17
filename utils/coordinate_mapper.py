"""Coordinate mapping helpers for LILA BLACK minimaps."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
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

MAP_IMAGE_LAYOUT: Dict[str, Dict[str, float]] = {
    "AmbroseValley": {"left_offset": 0.08, "top_offset": 0.09, "scale": 0.83},
    "GrandRift": {"left_offset": 0.09, "top_offset": 0.10, "scale": 0.83},
    "Lockdown": {"left_offset": 0.07, "top_offset": 0.07, "scale": 0.86},
}

PADDING_RATIO = 0.05


def _padded_bounds(map_id: str) -> Dict[str, float]:
    """Shrink raw bounds by 5% on each side to reduce outlier-driven stretch."""
    bounds = MAP_BOUNDS[map_id]
    x_range = bounds["x_max"] - bounds["x_min"]
    z_range = bounds["z_max"] - bounds["z_min"]

    return {
        "x_min": bounds["x_min"] + (x_range * PADDING_RATIO),
        "x_max": bounds["x_max"] - (x_range * PADDING_RATIO),
        "z_min": bounds["z_min"] + (z_range * PADDING_RATIO),
        "z_max": bounds["z_max"] - (z_range * PADDING_RATIO),
    }


def map_points_to_pixels(
    x_values: pd.Series,
    z_values: pd.Series,
    map_id: str,
    image_width: int,
    image_height: int,
) -> Tuple[pd.Series, pd.Series]:
    """Map game coordinates to minimap pixels with offsets and boundary clamping."""
    if map_id not in MAP_BOUNDS or map_id not in MAP_IMAGE_LAYOUT:
        raise ValueError(f"Unknown map_id: {map_id}")

    bounds = _padded_bounds(map_id)
    layout = MAP_IMAGE_LAYOUT[map_id]

    usable_width = image_width * layout["scale"]
    usable_height = image_height * layout["scale"]
    x_denom = bounds["x_max"] - bounds["x_min"]
    z_denom = bounds["z_max"] - bounds["z_min"]

    if x_denom == 0:
        x_denom = 1.0
    if z_denom == 0:
        z_denom = 1.0

    pixel_x = (
        layout["left_offset"] * image_width
        + ((x_values - bounds["x_min"]) / x_denom) * usable_width
    )

    pixel_z = (
        layout["top_offset"] * image_height
        + (1 - (z_values - bounds["z_min"]) / z_denom) * usable_height
    )

    # Clamp into the playable sub-rectangle (not full image) to keep points
    # out of black minimap padding margins.
    min_x = layout["left_offset"] * image_width
    max_x = min_x + usable_width
    min_z = layout["top_offset"] * image_height
    max_z = min_z + usable_height

    pixel_x = pixel_x.clip(lower=min_x, upper=max_x)
    pixel_z = pixel_z.clip(lower=min_z, upper=max_z)
    return pixel_x, pixel_z


def filter_position_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop extreme x/z outliers (2 std dev) before visualization."""
    if dataframe.empty or "x" not in dataframe.columns or "z" not in dataframe.columns:
        return dataframe

    x_mean = dataframe["x"].mean()
    x_std = dataframe["x"].std()
    z_mean = dataframe["z"].mean()
    z_std = dataframe["z"].std()

    if pd.isna(x_std) or pd.isna(z_std) or x_std == 0 or z_std == 0:
        return dataframe.copy()

    cleaned = dataframe[
        dataframe["x"].between(x_mean - 2 * x_std, x_mean + 2 * x_std)
        & dataframe["z"].between(z_mean - 2 * z_std, z_mean + 2 * z_std)
    ].copy()
    return cleaned


def clamp_pixels_to_non_black_mask(
    pixel_x: pd.Series,
    pixel_z: pd.Series,
    map_image: np.ndarray,
    threshold: int = 16,
) -> Tuple[pd.Series, pd.Series]:
    """
    Clamp pixel points into visible non-black minimap area.

    This handles irregular island shapes where rectangular clamping is not enough.
    """
    if pixel_x.empty or pixel_z.empty:
        return pixel_x, pixel_z

    if map_image.ndim != 3 or map_image.shape[2] < 3:
        return pixel_x, pixel_z

    height, width = map_image.shape[0], map_image.shape[1]
    mask = (
        (map_image[:, :, 0] > threshold)
        | (map_image[:, :, 1] > threshold)
        | (map_image[:, :, 2] > threshold)
    )

    if not mask.any():
        return pixel_x, pixel_z

    row_has = mask.any(axis=1)
    valid_rows = np.where(row_has)[0]
    if valid_rows.size == 0:
        return pixel_x, pixel_z

    min_x_by_row = np.where(row_has, np.argmax(mask, axis=1), 0)
    max_x_by_row = np.where(row_has, width - 1 - np.argmax(mask[:, ::-1], axis=1), width - 1)

    x_arr = pixel_x.to_numpy(dtype=float, copy=True)
    z_arr = pixel_z.to_numpy(dtype=float, copy=True)
    z_idx = np.rint(z_arr).astype(int)
    z_idx = np.clip(z_idx, 0, height - 1)

    # If a point is on a fully-black row, move it to nearest row with visible map pixels.
    pos = np.searchsorted(valid_rows, z_idx)
    left_idx = np.clip(pos - 1, 0, valid_rows.size - 1)
    right_idx = np.clip(pos, 0, valid_rows.size - 1)
    left_rows = valid_rows[left_idx]
    right_rows = valid_rows[right_idx]
    nearest_rows = np.where(
        np.abs(z_idx - left_rows) <= np.abs(right_rows - z_idx),
        left_rows,
        right_rows,
    )
    use_rows = np.where(row_has[z_idx], z_idx, nearest_rows)

    z_arr = np.where(row_has[z_idx], z_arr, use_rows.astype(float))
    x_arr = np.clip(x_arr, min_x_by_row[use_rows], max_x_by_row[use_rows])
    x_arr = np.clip(x_arr, 0, width - 1)
    z_arr = np.clip(z_arr, 0, height - 1)

    # Final local snap for any point still on black pixels.
    x_idx = np.rint(x_arr).astype(int)
    y_idx = np.rint(z_arr).astype(int)
    bad = ~mask[y_idx, x_idx]
    if bad.any():
        for i in np.where(bad)[0]:
            y0 = y_idx[i]
            x0 = x_idx[i]
            found = False
            for radius in range(1, 26):
                y_min = max(0, y0 - radius)
                y_max = min(height - 1, y0 + radius)
                x_min = max(0, x0 - radius)
                x_max = min(width - 1, x0 + radius)
                sub_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
                if not sub_mask.any():
                    continue
                ys, xs = np.where(sub_mask)
                ys = ys + y_min
                xs = xs + x_min
                dist = (ys - y0) ** 2 + (xs - x0) ** 2
                nearest = int(dist.argmin())
                z_arr[i] = float(ys[nearest])
                x_arr[i] = float(xs[nearest])
                found = True
                break
            if not found:
                x_arr[i] = float(np.clip(x_arr[i], 0, width - 1))
                z_arr[i] = float(np.clip(z_arr[i], 0, height - 1))

    return pd.Series(x_arr, index=pixel_x.index), pd.Series(z_arr, index=pixel_z.index)
