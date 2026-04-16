"""Data loading and filtering utilities for parquet telemetry."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

DATE_ORDER = ["February_10", "February_11", "February_12", "February_13", "February_14"]

EXPECTED_COLUMNS = [
    "date",
    "match_id",
    "user_id",
    "is_bot",
    "map_id",
    "x",
    "y",
    "z",
    "ts",
    "event",
    "source_file",
]


NON_PARQUET_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".ds_store",
}


def _normalize_data_dir(data_dir: str) -> str:
    """Handle nested structures like /player_data/player_data."""
    nested = os.path.join(data_dir, "player_data")
    if os.path.isdir(nested):
        return nested
    return data_dir


def _looks_like_data_file(file_name: str) -> bool:
    extension = os.path.splitext(file_name)[1].lower()
    if extension in NON_PARQUET_EXTENSIONS:
        return False
    if file_name.startswith("."):
        return False
    return True


@st.cache_data(show_spinner=False)
def list_data_files(data_dir: str) -> List[str]:
    """Find all potential parquet files in the data directory."""
    data_dir = _normalize_data_dir(data_dir)
    if not os.path.isdir(data_dir):
        return []

    files: List[str] = []
    for root, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if not _looks_like_data_file(file_name):
                continue
            files.append(os.path.join(root, file_name))

    files.sort()
    return files


@st.cache_data(show_spinner=False)
def load_parquet_file(file_path: str, columns: Tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read a parquet file defensively. Returns empty dataframe on failure."""
    use_columns = list(columns) if columns is not None else None

    try:
        dataframe = pd.read_parquet(file_path, columns=use_columns, engine="pyarrow")
        return _enrich_columns(dataframe, file_path)
    except Exception:
        try:
            # Retry full read because some files may miss requested columns.
            dataframe = pd.read_parquet(file_path, engine="pyarrow")
            if use_columns is not None:
                existing = [column for column in use_columns if column in dataframe.columns]
                dataframe = dataframe[existing]
            return _enrich_columns(dataframe, file_path)
        except Exception:
            return pd.DataFrame()


def _normalize_is_bot_value(value: object) -> object:
    """Normalize mixed bool/string values into booleans."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return pd.NA


def _infer_is_bot(frame: pd.DataFrame) -> pd.Series:
    """Infer is_bot when source data does not provide it explicitly."""
    inferred = pd.Series(pd.NA, index=frame.index, dtype="boolean")

    if "event" in frame.columns:
        inferred.loc[frame["event"] == "BotPosition"] = True
        inferred.loc[frame["event"] == "Position"] = False

    if {"user_id", "event"}.issubset(frame.columns):
        bot_user_map = (
            frame.assign(_bot_evt=frame["event"] == "BotPosition")
            .groupby("user_id")["_bot_evt"]
            .any()
            .to_dict()
        )
        inferred = inferred.mask(inferred.isna(), frame["user_id"].map(bot_user_map))

    return inferred.astype("boolean")


def _enrich_columns(dataframe: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Standardize schema across parquet variants."""
    if dataframe.empty:
        return dataframe

    frame = dataframe.copy()

    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")

    if "date" not in frame.columns or frame["date"].isna().all():
        frame["date"] = os.path.basename(os.path.dirname(file_path))

    if "source_file" not in frame.columns:
        frame["source_file"] = os.path.basename(file_path)
    else:
        frame["source_file"] = frame["source_file"].fillna(os.path.basename(file_path))

    if "is_bot" not in frame.columns:
        frame["is_bot"] = pd.NA

    frame["is_bot"] = frame["is_bot"].map(_normalize_is_bot_value).astype("boolean")

    inferred = _infer_is_bot(frame)
    frame["is_bot"] = frame["is_bot"].mask(frame["is_bot"].isna(), inferred)
    frame["is_bot"] = frame["is_bot"].fillna(False).astype("boolean")

    return frame


@st.cache_data(show_spinner=False)
def build_index(data_dir: str) -> pd.DataFrame:
    """Build index of available map/date/match combinations for filters."""
    index_rows: List[pd.DataFrame] = []

    for file_path in list_data_files(data_dir):
        frame = load_parquet_file(file_path, columns=("date", "map_id", "match_id"))
        if frame.empty:
            continue
        required = {"date", "map_id", "match_id"}
        if not required.issubset(frame.columns):
            continue

        frame = frame[list(required)].dropna().drop_duplicates()
        if not frame.empty:
            index_rows.append(frame)

    if not index_rows:
        return pd.DataFrame(columns=["date", "map_id", "match_id"])

    index_df = pd.concat(index_rows, ignore_index=True).drop_duplicates()
    return index_df


def strip_nakama_suffix(match_id: str) -> str:
    """Cleaner match ID for UI display."""
    if not isinstance(match_id, str):
        return ""
    return re.sub(r"\.nakama-\d+$", "", match_id)


def date_sort_key(value: str) -> int:
    if value in DATE_ORDER:
        return DATE_ORDER.index(value)
    return len(DATE_ORDER)


def get_available_dates(index_df: pd.DataFrame, map_id: str) -> List[str]:
    """Available dates for selected map."""
    if index_df.empty:
        return []
    dates = index_df[index_df["map_id"] == map_id]["date"].dropna().unique().tolist()
    dates.sort(key=date_sort_key)
    return dates


def get_matches(index_df: pd.DataFrame, map_id: str, date: str) -> List[str]:
    """Available match IDs for selected map/date."""
    if index_df.empty:
        return []

    rows = index_df[(index_df["map_id"] == map_id) & (index_df["date"] == date)]
    matches = rows["match_id"].dropna().unique().tolist()
    matches.sort()
    return matches


@st.cache_data(show_spinner=False)
def load_date_slice(data_dir: str, date: str) -> pd.DataFrame:
    """Load all files for a date folder into one dataframe (cached)."""
    data_dir = _normalize_data_dir(data_dir)
    date_dir = os.path.join(data_dir, date)

    candidate_files: List[str]
    if os.path.isdir(date_dir):
        candidate_files = [
            os.path.join(date_dir, file_name)
            for file_name in os.listdir(date_dir)
            if os.path.isfile(os.path.join(date_dir, file_name)) and _looks_like_data_file(file_name)
        ]
        candidate_files.sort()
    else:
        candidate_files = list_data_files(data_dir)

    chunks: List[pd.DataFrame] = []
    for file_path in candidate_files:
        frame = load_parquet_file(file_path, columns=tuple(EXPECTED_COLUMNS))
        if frame.empty:
            continue
        for column in EXPECTED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        chunks.append(frame[EXPECTED_COLUMNS])

    if not chunks:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    merged = pd.concat(chunks, ignore_index=True)
    merged["ts"] = pd.to_datetime(merged["ts"], errors="coerce")

    if "is_bot" in merged.columns:
        merged["is_bot"] = merged["is_bot"].astype("boolean")

    return merged


def load_filtered_match_data(data_dir: str, map_id: str, date: str, match_id: str) -> pd.DataFrame:
    """Load and filter to a single match selection."""
    data = load_date_slice(data_dir, date)
    if data.empty:
        return data

    filtered = data[
        (data["date"] == date)
        & (data["map_id"] == map_id)
        & (data["match_id"] == match_id)
    ].copy()

    filtered = filtered.sort_values("ts", kind="mergesort")
    return filtered.reset_index(drop=True)


def default_data_dir(project_root: str) -> str:
    """Best-effort default data directory for local and cloud usage."""
    repo_data = os.path.join(project_root, "data")
    home = os.path.expanduser("~")
    candidates = [
        repo_data,
        os.path.join(home, "Downloads", "player_data", "player_data"),
        os.path.join(home, "Downloads", "player_data"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate) and list_data_files(candidate):
            return candidate

    return repo_data


def build_match_display_map(match_ids: List[str]) -> Dict[str, str]:
    """Map compact display IDs back to original full IDs."""
    display_map: Dict[str, str] = {}
    for raw_id in match_ids:
        short_id = strip_nakama_suffix(raw_id)
        display_key = short_id
        if display_key in display_map and display_map[display_key] != raw_id:
            display_key = raw_id
        display_map[display_key] = raw_id
    return display_map
