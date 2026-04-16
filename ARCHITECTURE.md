# LILA Player Journey Tool - Architecture

## 1. Tech Stack Choices and Why

- **Streamlit**: Fast development for internal tooling with a simple filter-first UX suited to level designers.
- **Pandas + PyArrow**: Efficient parquet reading and dataframe-based transformations.
- **Plotly**: Interactive map overlays, distinct marker styling, and heatmap rendering in one charting system.
- **Pillow**: Lightweight minimap loading and conversion into arrays for Plotly background images.

## 2. Data Flow (Parquet to Screen)

1. **Data discovery** (`utils/data_loader.py`): recursively discovers candidate parquet files from a configured data directory.
2. **Index build** (`build_index`): cached extraction of `date`, `map_id`, and `match_id` for fast filter dropdowns.
3. **Match slice load** (`load_filtered_match_data`): cached loading of date-level parquet chunks, then filtering by map/date/match.
4. **User filters in sidebar**: map + date + match + player type + event type are applied on the same dataframe.
5. **Timeline cutoff**: chosen timestamp index restricts visible records to events up to that point.
6. **Visualization mode**:
- Path mode: render movement lines + event markers.
- Heatmap mode: render one overlay (kill/death/traffic/loot) at a time to avoid visual confusion.
7. **Stats panel**: computes player composition, kills, loot, storm deaths, and duration from selected match data.

## 3. Coordinate Mapping Approach (X/Z + Z Flip)

The telemetry includes `x`, `y`, and `z`, where `y` is elevation and not suitable for 2D minimap plotting. The app maps **X and Z** into pixel space using per-map bounds:

- `pixel_x = (game_x - x_min) / (x_max - x_min) * image_width`
- `pixel_z = (1 - (game_z - z_min) / (z_max - z_min)) * image_height`

The `1 - (...)` term flips Z because game coordinates are bottom-origin while image pixels are top-origin.

## 4. Assumptions

- `y` is elevation and intentionally ignored for 2D placement.
- Timestamps may resolve to 1970 epoch-based values; they are used only for **relative ordering/progression** inside a match.
- Matches can be bot-heavy; all metrics and visual states handle zero-human or zero-kill edge cases without crashing.
- Bot movement can appear as `BotPosition` (and occasionally `Position` with `is_bot=True`), both are supported for path rendering.

## 5. Tradeoffs

| Decision | Option A | Option B | Chose | Why |
|---|---|---|---|---|
| Data loading strategy | Load all parquet into memory once | Load by selected date then filter | Option B | Better memory behavior on large datasets while keeping UI responsive with caching |
| Interactive click handling | Native Streamlit chart only | Add `streamlit-plotly-events` | Option B | Enables click-to-highlight behavior expected by designers |
| Visual mode combination | Show paths + heatmaps together | Mode toggle between the two | Option B | Reduces clutter and preserves readability for non-data-science users |
| Timestamp presentation | Absolute timestamp labels | Relative match progression | Option B | Prevents confusion from epoch/1970 artifacts and matches design intent |
| Bot path source | Only `BotPosition` | `BotPosition` + bot `Position` fallback | Option B | More robust to telemetry variations across files |
