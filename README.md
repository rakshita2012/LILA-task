# LILA Games - Player Journey Visualization Tool

An interactive Streamlit tool for LILA BLACK level designers to inspect how players and bots move, fight, and loot across each map.

## Project Overview

This app lets non-technical game designers explore match telemetry on top of minimap images using a clean filter-driven interface. It supports map/date/match filtering, player path rendering, event overlays, heatmaps, playback over match timeline, and a compact stats panel.

## Tech Stack

- Python
- Streamlit (UI and app framework)
- Pandas + PyArrow (parquet ingestion and filtering)
- Plotly (interactive overlays and heatmaps)
- Pillow (minimap image loading)

## Repository Structure

- `app.py`: Main Streamlit application entry point
- `.streamlit/config.toml`: Streamlit dark theme config
- `utils/data_loader.py`: Cached parquet loading, indexing, and filtering
- `utils/coordinate_mapper.py`: X/Z to pixel conversion logic
- `utils/heatmap.py`: Heatmap event bucketing and overlay traces
- `maps/`: Minimap assets
- `data/`: Place parquet data files here (`.gitkeep` included)

## Environment Setup

1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run Locally

1. Clone this repository.
2. Ensure minimaps exist in `maps/`.
3. Put parquet files into `data/` or keep them in an external folder.
4. Run Streamlit:

```bash
streamlit run app.py
```

5. In the app sidebar, use the `Data folder` input if your parquet files are not under `data/`.

## How to Add Parquet Data Files

Option A:
- Copy parquet files into the repository `data/` folder (preserve date subfolders if available).

Option B:
- Keep data elsewhere and provide that absolute folder path in the app sidebar `Data folder` field.

Expected schema columns:
- `date, match_id, user_id, is_bot, map_id, x, y, z, ts, event, source_file`

## Streamlit Cloud Deployment

- Entry point: `app.py`
- Dependencies: pinned in `requirements.txt`
- Theme config: `.streamlit/config.toml`

After deploy, update this URL:
- **Deployed app:** [INSERT STREAMLIT CLOUD URL]
