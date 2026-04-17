"""LILA BLACK Mission Control dashboard."""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    plotly_events = None
    HAS_PLOTLY_EVENTS = False

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except Exception:
    gaussian_kde = None
    HAS_SCIPY = False

from utils.coordinate_mapper import (
    clamp_pixels_to_non_black_mask,
    filter_position_outliers,
    map_points_to_pixels,
)
from utils.data_loader import (
    build_index,
    build_match_display_map,
    default_data_dir,
    get_available_dates,
    get_matches,
    load_date_slice,
    load_filtered_match_data,
)
from utils.heatmap import build_heatmap_trace

st.set_page_config(page_title="LILA BLACK // Mission Control", page_icon="🎯", layout="wide")

PALETTE = {
    "bg": "#0A0A0F",
    "surface": "#12121A",
    "border": "#1E1E2E",
    "cyan": "#00F5FF",
    "red": "#FF3B3B",
    "amber": "#FFB800",
    "green": "#00FF88",
    "storm": "#9B59FF",
    "muted": "#6B7280",
    "text": "#E2E8F0",
}

MAP_OPTIONS = ["AmbroseValley", "GrandRift", "Lockdown"]
HEATMAPS = ["Kill heatmap", "Death heatmap", "Traffic heatmap", "Loot heatmap"]

EVENT_STYLE = {
    "BotKill": {"color": "#FFB800", "symbol": "triangle-up", "size": 11},
    "BotKilled": {"color": "#FFB800", "symbol": "triangle-down", "size": 11},
    "Kill": {"color": "#FF3B3B", "symbol": "circle", "size": 10},
    "Killed": {"color": "#8B1D1D", "symbol": "circle", "size": 9},
    "KilledByStorm": {"color": "#9B59FF", "symbol": "diamond", "size": 10},
    "Loot": {"color": "#FFB800", "symbol": "star", "size": 10},
}


def to_text(value) -> str:
    """Ensure dataframe labels are JSON-serializable strings."""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", errors="ignore")
    return str(value)


def css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
:root{{--bg:{PALETTE["bg"]};--surface:{PALETTE["surface"]};--border:{PALETTE["border"]};--cyan:{PALETTE["cyan"]};--red:{PALETTE["red"]};--amber:{PALETTE["amber"]};--storm:{PALETTE["storm"]};--green:{PALETTE["green"]};--muted:{PALETTE["muted"]};--text:{PALETTE["text"]};}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stAppViewContainer"]{{background:var(--bg);}}
[data-testid="stSidebar"]{{background:#0D0D14;border-right:1px solid var(--border);}}
.block-container{{max-width:100%;padding:.7rem 1.2rem;}}
.top{{background:var(--surface);border:1px solid var(--border);border-bottom:1px solid var(--cyan);border-radius:6px;padding:12px 16px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;}}
.title{{font:700 28px 'Rajdhani',sans-serif;color:var(--cyan);letter-spacing:.8px;}}
.sub{{font:14px 'Inter',sans-serif;color:var(--muted);}}
@keyframes pulse{{0%{{opacity:1}}50%{{opacity:.3}}100%{{opacity:1}}}}
.live{{font:600 14px 'Rajdhani',sans-serif;color:var(--green);border:1px solid var(--border);padding:4px 10px;border-radius:4px;}}
.dot{{animation:pulse 2s infinite;color:var(--green);}}
.logo{{font:700 18px 'Rajdhani',sans-serif;color:var(--cyan);letter-spacing:2px;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:10px;text-align:center;margin-bottom:8px;}}
.stitle{{font:600 16px 'Rajdhani',sans-serif;color:var(--cyan);letter-spacing:2px;text-transform:uppercase;}}
.fsec{{font:600 13px 'Rajdhani',sans-serif;color:var(--cyan);letter-spacing:1px;text-transform:uppercase;margin-top:8px;}}
.div{{height:1px;background:linear-gradient(90deg,var(--cyan),transparent);opacity:.45;margin:6px 0;}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:10px;}}
.scan{{background-image:repeating-linear-gradient(to bottom,rgba(0,245,255,.015)0,rgba(0,245,255,.015)1px,transparent 2px,transparent 4px);}}
.head{{font:600 16px 'Rajdhani',sans-serif;color:var(--cyan);text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;}}
.mapbox{{border:1px solid var(--cyan);border-radius:6px;box-shadow:0 0 14px rgba(0,245,255,.15);overflow:hidden;}}
.stat{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--cyan);border-radius:6px;padding:10px 12px;margin-bottom:7px;}}
.sl{{font:11px 'Inter',sans-serif;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}}
.sv{{font:700 30px 'JetBrains Mono',monospace;line-height:1.1;}}
.ss{{font:11px 'Inter',sans-serif;color:var(--muted);text-transform:uppercase;}}
.legend{{font:11px 'JetBrains Mono',monospace;color:var(--text);display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;}}
.empty{{padding:110px 20px;text-align:center;border:1px dashed var(--border);border-radius:6px;font:700 26px 'Rajdhani',sans-serif;color:var(--cyan);letter-spacing:2px;}}
.load{{font:13px 'JetBrains Mono',monospace;color:var(--cyan);}}
.load span{{animation:pulse 1.2s infinite;display:inline-block;}}
.load span:nth-child(2){{animation-delay:.2s}} .load span:nth-child(3){{animation-delay:.4s}}
.stButton>button{{border-radius:4px;border:1px solid var(--cyan);background:#0F1118;color:var(--cyan);font:600 13px 'Rajdhani',sans-serif;letter-spacing:1px;text-transform:uppercase;}}
.stButton>button:hover{{box-shadow:0 0 8px rgba(0,245,255,.25);}}
div[data-baseweb="select"]>div{{background:#0F1118!important;border-color:var(--border)!important;}}
div[data-baseweb="select"]>div:focus-within{{border-color:var(--cyan)!important;box-shadow:0 0 0 1px rgba(0,245,255,.25)!important;}}
[data-baseweb="tag"]{{background:rgba(0,245,255,.12)!important;border:1px solid rgba(0,245,255,.35)!important;color:var(--text)!important;}}
.stSlider [role="slider"]{{background:var(--cyan)!important;border:2px solid var(--bg)!important;}}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def minimap_bytes(root: str, map_id: str) -> bytes:
    with open(os.path.join(root, "maps", f"{map_id}.png"), "rb") as f:
        return f.read()


def minimap_img(root: str, map_id: str) -> Image.Image:
    return Image.open(io.BytesIO(minimap_bytes(root, map_id))).convert("RGB")


def add_bg(fig: go.Figure, img: Image.Image, w: int, h: int) -> None:
    fig.add_layout_image(dict(source=img, x=0, y=h, sizex=w, sizey=h, xref="x", yref="y", xanchor="left", yanchor="bottom", layer="below", sizing="stretch"))
    fig.update_xaxes(range=[0, w], visible=False, showgrid=False)
    fig.update_yaxes(range=[h, 0], visible=False, showgrid=False, scaleanchor="x", scaleratio=1)


def short_user(uid: str) -> str:
    return uid[:8] if isinstance(uid, str) else "unknown"


def filter_player(df, mode: str):
    if mode == "Humans only":
        return df[df["is_bot"] == False]
    if mode == "Bots only":
        return df[df["is_bot"] == True]
    return df


def stats(df):
    if df.empty:
        return {"pt": 0, "ph": 0, "pb": 0, "kt": 0, "kh": 0, "kb": 0, "loot": 0, "storm": 0, "dur": 0}
    users = df[["user_id", "is_bot"]].drop_duplicates()
    ph = int((users["is_bot"] == False).sum())
    pb = int((users["is_bot"] == True).sum())
    kills = df[df["event"].isin(["Kill", "BotKill"])]
    kh = int((kills["is_bot"] == False).sum())
    kb = int((kills["is_bot"] == True).sum())
    t = df["ts"].dropna()
    dur = int((t.max() - t.min()).total_seconds()) if not t.empty else 0
    return {"pt": ph + pb, "ph": ph, "pb": pb, "kt": len(kills), "kh": kh, "kb": kb, "loot": int((df["event"] == "Loot").sum()), "storm": int((df["event"] == "KilledByStorm").sum()), "dur": max(dur, 0)}


def render_cards(stv, animate: bool):
    p = st.empty()
    steps = 10 if animate else 1
    for i in range(1, steps + 1):
        r = i / steps
        m, s = divmod(stv["dur"], 60)
        p.markdown(
            f"""
<div class="head">// MATCH INTEL</div><div class="div"></div>
<div class="stat" style="border-left-color:{PALETTE["cyan"]}"><div class="sl">Players</div><div class="sv" style="color:{PALETTE["cyan"]}">{int(stv["pt"]*r)}</div><div class="ss">humans {stv["ph"]} | bots {stv["pb"]}</div></div>
<div class="stat" style="border-left-color:{PALETTE["red"]}"><div class="sl">Kills</div><div class="sv" style="color:{PALETTE["red"]}">{int(stv["kt"]*r)}</div><div class="ss">human {stv["kh"]} | bot {stv["kb"]}</div></div>
<div class="stat" style="border-left-color:{PALETTE["amber"]}"><div class="sl">Loot</div><div class="sv" style="color:{PALETTE["amber"]}">{int(stv["loot"]*r)}</div><div class="ss">resource pickups</div></div>
<div class="stat" style="border-left-color:{PALETTE["storm"]}"><div class="sl">Storm Deaths</div><div class="sv" style="color:{PALETTE["storm"]}">{int(stv["storm"]*r)}</div><div class="ss">zone pressure</div></div>
<div class="stat" style="border-left-color:{PALETTE["green"]}"><div class="sl">Duration</div><div class="sv" style="font-size:24px;color:{PALETTE["green"]}">{m:02d}:{s:02d}</div><div class="ss">mm:ss</div></div>
""",
            unsafe_allow_html=True,
        )
        if animate:
            time.sleep(0.05)


def event_bar(df) -> go.Figure:
    c = df["event"].map(to_text).value_counts().sort_values(ascending=True)
    if c.empty:
        c = c.reindex(["No Data"]).fillna(0)
    cols = []
    for e in c.index:
        if e in EVENT_STYLE:
            cols.append(EVENT_STYLE[e]["color"])
        elif e == "Position":
            cols.append(PALETTE["cyan"])
        elif e == "BotPosition":
            cols.append("#5AA9E6")
        else:
            cols.append(PALETTE["muted"])
    fig = go.Figure([go.Bar(x=c.values.tolist(), y=c.index.tolist(), orientation="h", marker={"color": cols}, text=c.values.tolist(), textposition="outside", hovertemplate="%{y}: %{x}<extra></extra>")])
    fig.update_layout(paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], font={"color": PALETTE["text"], "family": "Inter"}, margin={"l": 100, "r": 8, "t": 6, "b": 8}, height=230, xaxis={"visible": False}, yaxis={"showgrid": False})
    return fig


@st.cache_data(show_spinner=False)
def load_map_intelligence_slice(data_dir: str, date: str, map_id: str) -> pd.DataFrame:
    df = load_date_slice(data_dir, date)
    if df.empty:
        return df
    for col in ["date", "match_id", "user_id", "map_id", "event"]:
        if col in df.columns:
            df[col] = df[col].map(to_text)
    return df[df["map_id"] == map_id].copy()


def avg_match_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"avg_players": 0.0, "avg_kills": 0.0, "avg_human_kills": 0.0, "avg_loot": 0.0, "avg_storm": 0.0, "total_matches": 0}
    per_match = df.groupby("match_id").agg(
        players=("user_id", "nunique"),
        kills=("event", lambda s: int(s.isin(["Kill", "BotKill"]).sum())),
        human_kills=("event", lambda s: int((s == "Kill").sum())),
        loot=("event", lambda s: int((s == "Loot").sum())),
        storm=("event", lambda s: int((s == "KilledByStorm").sum())),
    )
    return {
        "avg_players": float(per_match["players"].mean()),
        "avg_kills": float(per_match["kills"].mean()),
        "avg_human_kills": float(per_match["human_kills"].mean()),
        "avg_loot": float(per_match["loot"].mean()),
        "avg_storm": float(per_match["storm"].mean()),
        "total_matches": int(per_match.shape[0]),
    }


def zone_frame(df: pd.DataFrame, map_id: str, img_w: int, img_h: int, map_np: np.ndarray, events: list[str] | None = None) -> pd.DataFrame:
    f = df.copy()
    if events is not None:
        f = f[f["event"].isin(events)].copy()
    if f.empty:
        return f
    f["px"], f["pz"] = map_points_to_pixels(f["x"], f["z"], map_id, img_w, img_h)
    f["px"], f["pz"] = clamp_pixels_to_non_black_mask(f["px"], f["pz"], map_np)
    f["grid_col"] = np.clip((f["px"] / img_w * 4).astype(int), 0, 3)
    f["grid_row"] = np.clip((f["pz"] / img_h * 4).astype(int), 0, 3)
    rows = ["Top", "Upper-Mid", "Lower-Mid", "Bottom"]
    cols = ["Left", "Left-Center", "Right-Center", "Right"]
    f["zone"] = f["grid_row"].map(lambda i: rows[int(i)]) + "-" + f["grid_col"].map(lambda i: cols[int(i)])
    return f


def classify_zone(pct: float) -> tuple[str, str]:
    if pct < 2:
        return "DEAD ZONE", PALETTE["red"]
    if pct < 8:
        return "LOW ACTIVITY", PALETTE["amber"]
    if pct <= 15:
        return "ACTIVE", PALETTE["green"]
    return "HOTSPOT", PALETTE["cyan"]


def rec_card(text: str) -> str:
    return (
        "<div style='background:#12121A;border:1px solid #1E1E2E;border-left:3px solid #FFB800;"
        "border-radius:6px;padding:10px 12px;margin-bottom:8px;color:#E2E8F0;font-family:Rajdhani,sans-serif;'>"
        f"INFO {text}</div>"
    )

def ensure_state():
    defaults = {"timeline_idx": 0, "playing": False, "play_speed": 1.0, "highlight_user": None, "view_mode": "Paths + Events", "heatmap_type": "Traffic heatmap", "last_anim_match": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    css()
    ensure_state()
    root = os.path.dirname(os.path.abspath(__file__))
    st.markdown('<div class="top"><div><div class="title">◈ LILA BLACK  //  MISSION CONTROL</div><div class="sub">Player Journey Intelligence System</div></div><div class="live">LIVE <span class="dot">●</span></div></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="logo">LILA GAMES</div><div class="stitle">// FILTERS</div>', unsafe_allow_html=True)
        st.markdown('<div class="fsec">Map Select</div>', unsafe_allow_html=True)
        map_id = st.selectbox("Map", MAP_OPTIONS, label_visibility="collapsed")
        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        with st.expander("DATA SOURCE", expanded=False):
            data_dir = st.text_input("Data folder", value=default_data_dir(root))
        if not data_dir:
            data_dir = default_data_dir(root)
        idx = build_index(data_dir)
        dates = get_available_dates(idx, map_id)
        if not dates:
            st.error("No dates found for map.")
            st.stop()
        st.markdown('<div class="fsec">Date</div>', unsafe_allow_html=True)
        date = st.selectbox("Date", dates, label_visibility="collapsed")
        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        mids = get_matches(idx, map_id, date)
        dmap = build_match_display_map(mids)
        labels = ["-- SELECT MATCH --"] + sorted(dmap.keys())
        st.markdown('<div class="fsec">Match</div>', unsafe_allow_html=True)
        label = st.selectbox("Match", labels, label_visibility="collapsed")
        match_id = dmap.get(label)
        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="fsec">Players</div>', unsafe_allow_html=True)
        ptype = st.radio("Players", ["All", "Humans only", "Bots only"], label_visibility="collapsed")
        st.markdown('<div class="div"></div><div class="fsec">Events</div>', unsafe_allow_html=True)

    if match_id is None:
        l, r = st.columns([1.85, 1], gap="large")
        with l:
            st.markdown('<div class="panel scan"><div class="empty">⌖ SELECT A MATCH TO BEGIN ANALYSIS</div></div>', unsafe_allow_html=True)
        with r:
            st.markdown('<div class="panel"><div class="head">// MATCH INTEL</div><p style="color:#6B7280">Select a match to activate telemetry.</p></div>', unsafe_allow_html=True)
        return

    loading = st.empty()
    loading.markdown('<div class="load">// LOADING MATCH DATA<span>.</span><span>.</span><span>.</span></div>', unsafe_allow_html=True)
    mdf = load_filtered_match_data(data_dir, map_id, date, match_id)
    for col in ["date", "match_id", "user_id", "map_id", "event"]:
        if col in mdf.columns:
            mdf[col] = mdf[col].map(to_text)
    loading.empty()
    if mdf.empty:
        st.warning("No records found for selected match.")
        return

    ev = sorted(mdf["event"].dropna().unique().tolist())
    with st.sidebar:
        selected_events = st.multiselect("Event types", ev, default=ev, label_visibility="collapsed")

    fdf = filter_player(mdf, ptype)
    fdf = fdf[fdf["event"].isin(selected_events)] if selected_events else fdf.iloc[0:0]
    tsource = filter_player(mdf, ptype)
    ts_vals = sorted([t for t in tsource["ts"].dropna().unique()])
    if ts_vals:
        st.session_state["timeline_idx"] = min(st.session_state["timeline_idx"], len(ts_vals) - 1)
        cutoff = ts_vals[st.session_state["timeline_idx"]]
        vdf = fdf[fdf["ts"] <= cutoff].copy()
    else:
        vdf = fdf.copy()

    left, right = st.columns([1.85, 1], gap="large")
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        stv = stats(mdf)
        render_cards(stv, st.session_state["last_anim_match"] != match_id)
        st.session_state["last_anim_match"] = match_id
        st.markdown('<div class="head">// EVENT BREAKDOWN</div>', unsafe_allow_html=True)
        st.plotly_chart(event_bar(vdf if not vdf.empty else mdf), use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["zoom", "pan", "select", "lasso2d", "autoScale", "resetScale"]})
        st.markdown('<div class="head">// VIEW MODE</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("PATHS", use_container_width=True):
                st.session_state["view_mode"] = "Paths + Events"
        with c2:
            if st.button("HEATMAP", use_container_width=True):
                st.session_state["view_mode"] = "Heatmap"
        hcols = st.columns(4)
        for i, hm in enumerate(HEATMAPS):
            with hcols[i]:
                if st.button(hm.split()[0].upper(), key=f"hm_{hm}", use_container_width=True):
                    st.session_state["view_mode"] = "Heatmap"
                    st.session_state["heatmap_type"] = hm
        st.caption(f"Active: {st.session_state['view_mode']} | {st.session_state['heatmap_type']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        st.markdown('<div class="panel scan">', unsafe_allow_html=True)
        st.markdown(f'<div class="head">// TACTICAL MAP :: {map_id}</div>', unsafe_allow_html=True)
        img = minimap_img(root, map_id)
        w, h = img.size
        fig = go.Figure()
        add_bg(fig, img, w, h)

        if st.session_state["view_mode"] == "Heatmap":
            hm = build_heatmap_trace(vdf, map_id, w, h, st.session_state["heatmap_type"])
            if hm is not None:
                fig.add_trace(hm)
        else:
            paths = vdf[((vdf["event"] == "Position") & (vdf["is_bot"] == False)) | ((vdf["event"].isin(["BotPosition", "Position"])) & (vdf["is_bot"] == True))].copy()
            em = vdf[vdf["event"].isin(EVENT_STYLE.keys())].copy()
            if not paths.empty:
                paths["px"], paths["pz"] = map_points_to_pixels(paths["x"], paths["z"], map_id, w, h)
                users = paths["user_id"].dropna().unique().tolist()
                hi = st.session_state.get("highlight_user")
                for u in users:
                    up = paths[paths["user_id"] == u].sort_values("ts")
                    is_bot = bool(up["is_bot"].iloc[0])
                    op = 0.75 if is_bot else 1.0
                    if hi and hi != u:
                        op = 0.1
                    if is_bot:
                        outer_color = "#FF6B35"
                        inner_color = "#FF6B35"
                        outer_width = 4
                        inner_width = 1.7
                        dash_style = "dot"
                        outer_opacity = 0.24
                        inner_opacity = 0.75
                    else:
                        outer_color = "#00F5FF"
                        inner_color = "#FFFFFF"
                        outer_width = 5
                        inner_width = 2.4
                        dash_style = "solid"
                        outer_opacity = 0.28
                        inner_opacity = 1.0

                    if hi and hi == u:
                        outer_width += 1.2
                        inner_width += 0.5

                    # Glow layer for contrast on dark/teal map regions.
                    fig.add_trace(
                        go.Scattergl(
                            x=up["px"],
                            y=up["pz"],
                            mode="lines",
                            line={"color": outer_color, "width": outer_width, "dash": dash_style},
                            opacity=outer_opacity * op,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    # Core path layer.
                    fig.add_trace(
                        go.Scattergl(
                            x=up["px"],
                            y=up["pz"],
                            mode="lines",
                            line={"color": inner_color, "width": inner_width, "dash": dash_style},
                            opacity=inner_opacity * op,
                            name=f"{short_user(u)} ({'BOT' if is_bot else 'HUM'})",
                            customdata=[[u]] * len(up),
                            text=[short_user(u)] * len(up),
                            hovertemplate="User: %{text}<br>Event: Path<br>x=%{x:.1f}, z=%{y:.1f}<extra></extra>",
                        )
                    )
            if not em.empty:
                em["px"], em["pz"] = map_points_to_pixels(em["x"], em["z"], map_id, w, h)
                for en, stl in EVENT_STYLE.items():
                    ss = em[em["event"] == en]
                    if ss.empty:
                        continue
                    fig.add_trace(go.Scattergl(x=ss["px"], y=ss["pz"], mode="markers", marker={"color": stl["color"], "symbol": stl["symbol"], "size": stl["size"], "line": {"width": .5, "color": "#111"}}, name=en, text=[short_user(x) for x in ss["user_id"].tolist()], customdata=ss["ts"].astype(str).tolist(), hovertemplate=f"Event: {en}<br>User: %{{text}}<br>Timestamp: %{{customdata}}<br>x=%{{x:.1f}}, z=%{{y:.1f}}<extra></extra>"))

        fig.update_layout(height=760, margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], font={"color": PALETTE["text"], "family": "Inter"}, legend={"orientation": "h", "y": -0.06, "font": {"size": 10}})
        st.markdown('<div class="mapbox">', unsafe_allow_html=True)
        pts = []
        if HAS_PLOTLY_EVENTS:
            pts = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=760,
                override_width="100%",
                key="tact_map",
            )
        else:
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown("</div>", unsafe_allow_html=True)

        if pts and st.session_state["view_mode"] == "Paths + Events":
            c = pts[0]
            cn, pn = c.get("curveNumber"), c.get("pointNumber")
            if cn is not None and pn is not None:
                tr = fig.data[cn]
                if hasattr(tr, "customdata") and tr.customdata is not None:
                    picked = tr.customdata[pn]
                    if isinstance(picked, (list, tuple)) and picked:
                        st.session_state["highlight_user"] = picked[0]
                        st.rerun()

        st.markdown(f'<div class="legend"><span style="color:{PALETTE["amber"]}">●</span>BOTKILL <span style="color:{PALETTE["red"]}">●</span>KILL <span style="color:#8B1D1D">●</span>DEATH <span style="color:{PALETTE["storm"]}">●</span>STORM <span style="color:{PALETTE["amber"]}">●</span>LOOT</div>', unsafe_allow_html=True)
        if st.session_state["view_mode"] == "Paths + Events":
            a, b = st.columns([1, 3])
            with a:
                if st.button("CLEAR HIGHLIGHT", use_container_width=True):
                    st.session_state["highlight_user"] = None
                    st.rerun()
            with b:
                if HAS_PLOTLY_EVENTS:
                    st.caption(f"Highlighted: `{short_user(st.session_state['highlight_user'])}`" if st.session_state.get("highlight_user") else "Click path line to focus one player.")
                else:
                    st.caption("Interactive click-highlighting is unavailable in this environment, but all paths/events remain visible.")

        if ts_vals:
            m = len(ts_vals) - 1
            t1, t2, t3 = st.columns([3, 1, 1])
            with t1:
                st.session_state["timeline_idx"] = st.slider("Mission timeline", 0, m, st.session_state["timeline_idx"], key="timeline")
            with t2:
                if st.button("PAUSE" if st.session_state["playing"] else "PLAY", use_container_width=True):
                    st.session_state["playing"] = not st.session_state["playing"]
            with t3:
                sp = st.selectbox("Speed", [0.5, 1.0, 2.0], index=[0.5, 1.0, 2.0].index(st.session_state["play_speed"]))
                st.session_state["play_speed"] = float(sp)
            base = ts_vals[0]
            now = ts_vals[st.session_state["timeline_idx"]]
            st.caption(f"Mission time: `{max((now-base).total_seconds(),0):.2f}s`")
        st.markdown("</div>", unsafe_allow_html=True)

    tab_map_label, tab_intel = st.tabs(["// TACTICAL MAP", "// MAP INTELLIGENCE"])
    with tab_map_label:
        st.caption("Tactical map is displayed above in the primary dashboard view.")
    with tab_intel:
        st.markdown('<div class="head">// MAP INTELLIGENCE</div>', unsafe_allow_html=True)
        ia, ib, ic = st.columns(3)
        with ia:
            intel_map = st.selectbox("Map selector", MAP_OPTIONS, index=MAP_OPTIONS.index(map_id), key="intel_map")
        intel_dates = get_available_dates(idx, intel_map)
        with ib:
            intel_date = st.selectbox("Date selector", intel_dates, index=(intel_dates.index(date) if date in intel_dates else 0), key="intel_date")
        with ic:
            intel_player = st.radio("Player type", ["All", "Humans only", "Bots only"], horizontal=True, key="intel_ptype")

        with st.spinner("// ANALYZING MAP TERRITORY..."):
            intel_df = load_map_intelligence_slice(data_dir, intel_date, intel_map)
            intel_df = filter_player(intel_df, intel_player)
            intel_df = filter_position_outliers(intel_df)

        if intel_df.empty:
            st.info("No telemetry found for the selected date/map/player filters.")
        else:
            mstats = avg_match_stats(intel_df)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("AVG PLAYERS PER MATCH", f"{mstats['avg_players']:.1f}")
            c2.metric("AVG KILLS PER MATCH", f"{mstats['avg_kills']:.1f}")
            c3.metric("AVG HUMAN KILLS PER MATCH", f"{mstats['avg_human_kills']:.1f}")
            c4.metric("AVG LOOT PICKUPS", f"{mstats['avg_loot']:.1f}")
            c5.metric("AVG STORM DEATHS", f"{mstats['avg_storm']:.1f}")
            c6.metric("TOTAL MATCHES ANALYZED", f"{mstats['total_matches']}")

            st.markdown('<div class="head">// TERRITORY COVERAGE - WHERE PLAYERS GO</div>', unsafe_allow_html=True)
            intel_img = minimap_img(root, intel_map)
            intel_np = np.array(intel_img)
            iw, ih = intel_img.size
            pos_df = zone_frame(intel_df, intel_map, iw, ih, intel_np, events=["Position", "BotPosition"])
            kde_fig = go.Figure()
            add_bg(kde_fig, intel_img, iw, ih)
            if not pos_df.empty:
                coords = pos_df[["px", "pz"]].dropna()
                if len(coords) > 10000:
                    coords = coords.sample(n=10000, random_state=42)
                if len(coords) >= 20:
                    xi = np.linspace(0, iw - 1, 140)
                    yi = np.linspace(0, ih - 1, 140)
                    if HAS_SCIPY:
                        xy = np.vstack([coords["px"].to_numpy(), coords["pz"].to_numpy()])
                        xx, yy = np.meshgrid(xi, yi)
                        kde = gaussian_kde(xy)
                        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                        zz = zz / (zz.max() if zz.max() > 0 else 1)
                    else:
                        hist, xedges, yedges = np.histogram2d(
                            coords["px"].to_numpy(),
                            coords["pz"].to_numpy(),
                            bins=140,
                            range=[[0, iw], [0, ih]],
                        )
                        zz = hist.T
                        zz = zz / (zz.max() if zz.max() > 0 else 1)
                        xi = (xedges[:-1] + xedges[1:]) / 2
                        yi = (yedges[:-1] + yedges[1:]) / 2
                    kde_fig.add_trace(
                        go.Heatmap(
                            x=xi,
                            y=yi,
                            z=zz,
                            colorscale=[[0.0, "#0A0A2A"], [0.35, "#00F5FF"], [0.7, "#FFB800"], [1.0, "#FF3B3B"]],
                            opacity=0.6,
                            showscale=False,
                        )
                    )
            kde_fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], font={"color": PALETTE["text"]})
            st.plotly_chart(kde_fig, use_container_width=True, config={"displaylogo": False})

            st.markdown('<div class="head">// DEAD ZONE ANALYSIS</div>', unsafe_allow_html=True)
            if pos_df.empty:
                st.info("Not enough position events for grid analysis.")
            else:
                zone_counts = pos_df["zone"].value_counts()
                total = max(int(zone_counts.sum()), 1)
                rows = []
                for zone_name, count in zone_counts.items():
                    pct = (count / total) * 100.0
                    cls, color = classify_zone(pct)
                    rows.append({"zone": zone_name, "pct": pct, "classification": cls, "color": color})
                zone_stats = pd.DataFrame(rows).sort_values("pct", ascending=True)
                table_html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'><thead><tr><th style='text-align:left;padding:6px;border-bottom:1px solid #1E1E2E'>Zone</th><th style='text-align:left;padding:6px;border-bottom:1px solid #1E1E2E'>% of Traffic</th><th style='text-align:left;padding:6px;border-bottom:1px solid #1E1E2E'>Classification</th></tr></thead><tbody>"
                for _, row in zone_stats.iterrows():
                    table_html += f"<tr><td>{row['zone']}</td><td>{row['pct']:.2f}%</td><td style='color:{row['color']}'>{row['classification']}</td></tr>"
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)

                st.markdown('<div class="head">// LOOT CONCENTRATION</div>', unsafe_allow_html=True)
                loot_df = zone_frame(intel_df, intel_map, iw, ih, intel_np, events=["Loot"])
                loot_zone = loot_df["zone"].value_counts().rename("loot_count") if not loot_df.empty else pd.Series(dtype=float, name="loot_count")
                traffic_zone = zone_counts.rename("traffic_count")
                comp = pd.concat([traffic_zone, loot_zone], axis=1).fillna(0)
                comp["traffic_pct"] = comp["traffic_count"] / max(comp["traffic_count"].sum(), 1) * 100.0
                comp["loot_pct"] = comp["loot_count"] / max(comp["loot_count"].sum(), 1) * 100.0
                comp = comp.sort_values("loot_count", ascending=False)

                l1, l2 = st.columns(2)
                with l1:
                    top5 = comp.head(5).sort_values("loot_count", ascending=True)
                    fig_l = go.Figure([go.Bar(x=top5["loot_count"].tolist(), y=top5.index.tolist(), orientation="h", marker={"color": PALETTE["amber"]})])
                    fig_l.update_layout(height=260, paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], font={"color": PALETTE["text"]}, margin={"l": 110, "r": 10, "t": 10, "b": 20}, title="Top 5 Most Looted Areas")
                    st.plotly_chart(fig_l, use_container_width=True, config={"displaylogo": False})
                with l2:
                    cmp = comp.sort_values("traffic_pct", ascending=False).head(8)
                    fig_c = go.Figure()
                    fig_c.add_bar(name="Traffic %", x=cmp.index.tolist(), y=cmp["traffic_pct"].tolist(), marker_color=PALETTE["cyan"])
                    fig_c.add_bar(name="Loot %", x=cmp.index.tolist(), y=cmp["loot_pct"].tolist(), marker_color=PALETTE["amber"])
                    fig_c.update_layout(barmode="group", height=260, paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], font={"color": PALETTE["text"]}, margin={"l": 10, "r": 10, "t": 10, "b": 60}, title="Loot Density vs Traffic Density")
                    st.plotly_chart(fig_c, use_container_width=True, config={"displaylogo": False})

                under = comp[(comp["loot_pct"] > comp["loot_pct"].median()) & (comp["traffic_pct"] < 2)]
                st.markdown('<div class="head">// ACTIONABLE RECOMMENDATIONS</div>', unsafe_allow_html=True)
                recs = []
                for _, row in zone_stats[zone_stats["pct"] < 2].iterrows():
                    recs.append(f"Zone {row['zone']} receives only {row['pct']:.2f}% of player traffic. Consider adding loot incentives, cover structures, or adjusting storm path to route players through this area.")
                for zname, row in under.iterrows():
                    recs.append(f"High loot density in {zname} but only {row['traffic_pct']:.2f}% player traffic. Loot placement may not be attracting players - consider adding visual signposting or improving access routes.")
                if not recs:
                    recs.append("Current map usage looks balanced for the selected filters. Focus on micro-adjustments near mid-traffic routes.")
                for rec in recs:
                    st.markdown(rec_card(rec), unsafe_allow_html=True)

    if ts_vals and st.session_state.get("playing"):
        if st.session_state["timeline_idx"] < len(ts_vals) - 1:
            time.sleep({0.5: 0.8, 1.0: 0.45, 2.0: 0.2}.get(st.session_state["play_speed"], 0.45))
            st.session_state["timeline_idx"] += 1
            st.rerun()
        else:
            st.session_state["playing"] = False


if __name__ == "__main__":
    main()
