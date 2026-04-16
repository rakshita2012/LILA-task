"""LILA BLACK Mission Control dashboard."""

from __future__ import annotations

import io
import os
import time

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
                colors = {u: ["#00F5FF", "#4CC9F0", "#00FF88", "#FFB800", "#FF3B3B", "#9B59FF", "#E2E8F0"][i % 7] for i, u in enumerate(sorted(users))}
                hi = st.session_state.get("highlight_user")
                for u in users:
                    up = paths[paths["user_id"] == u].sort_values("ts")
                    is_bot = bool(up["is_bot"].iloc[0])
                    op = 0.7 if is_bot else 1.0
                    if hi and hi != u:
                        op = 0.1
                    lw = 2.6 if hi and hi == u else 1.5
                    fig.add_trace(go.Scattergl(x=up["px"], y=up["pz"], mode="lines", line={"color": colors[u], "width": lw, "dash": "dash" if is_bot else "solid"}, opacity=op, name=f"{short_user(u)} ({'BOT' if is_bot else 'HUM'})", customdata=[[u]] * len(up), text=[short_user(u)] * len(up), hovertemplate="User: %{text}<br>Event: Path<br>x=%{x:.1f}, z=%{y:.1f}<extra></extra>"))
            if not em.empty:
                em["px"], em["pz"] = map_points_to_pixels(em["x"], em["z"], map_id, w, h)
                for en, stl in EVENT_STYLE.items():
                    ss = em[em["event"] == en]
                    if ss.empty:
                        continue
                    fig.add_trace(go.Scattergl(x=ss["px"], y=ss["pz"], mode="markers", marker={"color": stl["color"], "symbol": stl["symbol"], "size": stl["size"], "line": {"width": .5, "color": "#111"}}, name=en, text=[short_user(x) for x in ss["user_id"].tolist()], customdata=ss["ts"].astype(str).tolist(), hovertemplate=f"Event: {en}<br>User: %{{text}}<br>Timestamp: %{{customdata}}<br>x=%{{x:.1f}}, z=%{{y:.1f}}<extra></extra>"))

        fig.update_layout(height=760, margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], font={"color": PALETTE["text"], "family": "Inter"}, legend={"orientation": "h", "y": -0.06, "font": {"size": 10}})
        st.markdown('<div class="mapbox">', unsafe_allow_html=True)
        pts = plotly_events(fig, click_event=True, select_event=False, hover_event=False, override_height=760, override_width="100%", key="tact_map")
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
                st.caption(f"Highlighted: `{short_user(st.session_state['highlight_user'])}`" if st.session_state.get("highlight_user") else "Click path line to focus one player.")

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

    if ts_vals and st.session_state.get("playing"):
        if st.session_state["timeline_idx"] < len(ts_vals) - 1:
            time.sleep({0.5: 0.8, 1.0: 0.45, 2.0: 0.2}.get(st.session_state["play_speed"], 0.45))
            st.session_state["timeline_idx"] += 1
            st.rerun()
        else:
            st.session_state["playing"] = False


if __name__ == "__main__":
    main()
