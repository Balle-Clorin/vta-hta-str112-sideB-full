"""
app.py  —  CBS STR-112 Side B  ·  VTA & HTA vs Groove Radius  (Streamlit)
=========================================================================
Web version of str112_sideb_radius_plot.py.

Upload one WAV of the whole of Side B. The app

  1. detects the individual test-tone tracks (4 kHz carrier + 200/400 Hz tone,
     lateral = L/R in phase, vertical = L/R anti-phase),
  2. labels them Group A-D and +6 ... +15 dB from the STR-112 sequence,
  3. looks up peak velocity and groove radius from White & Gust (1982) Table 2,
  4. runs vta_analyzer_core.analyse() on the L and R channel of every track,
  5. plots HTA (lateral) and VTA (vertical) against groove radius.

Requires vta_analyzer_core.py (REV-23 or later) in the same folder as app.py.
"""

APP_REV    = "REV2"
APP_SCRIPT = "app.py"

import gc
import hashlib
import importlib.util
import io
import os
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent

# ── Colour palette (same dark theme as the desktop script) ───────────────────
DARK, MID, ACCENT, GREEN = "#0e1117", "#1c2230", "#00d4ff", "#39ff14"
AMBER, RED, WHITE, GREY = "#ffb300", "#ff3d3d", "#e8eaf0", "#5a6070"
PURPLE, ORANGE = "#c084fc", "#fb923c"

# ── White & Gust Table 2: (f_mod, is_lateral, band) -> (peak vel m/s, radius m)
GUST_TABLE = {
    (400, True,  "+6dB"):  (0.0563,  0.138),
    (400, True,  "+9dB"):  (0.0796,  0.135),
    (400, True,  "+12dB"): (0.1126,  0.132),
    (400, True,  "+15dB"): (0.1592,  0.128),
    (400, False, "+6dB"):  (0.0563,  0.114),
    (400, False, "+9dB"):  (0.0796,  0.110),
    (400, False, "+12dB"): (0.1126,  0.105),
    (200, True,  "+6dB"):  (0.02815, 0.100),
    (200, True,  "+9dB"):  (0.03980, 0.097),
    (200, True,  "+12dB"): (0.05630, 0.094),
    (200, True,  "+15dB"): (0.07960, 0.089),
    (200, False, "+6dB"):  (0.02815, 0.077),
    (200, False, "+9dB"):  (0.03980, 0.072),
    (200, False, "+12dB"): (0.05630, 0.067),
}

GROUP_DEFS = [          # label, mod_type, is_lateral, f_mod, n_bands
    ("A", "LAT", True,  400, 4),
    ("B", "VER", False, 400, 3),
    ("C", "LAT", True,  200, 4),
    ("D", "VER", False, 200, 3),
]
BAND_LABELS = ["+6dB", "+9dB", "+12dB", "+15dB"]
GROUP_COLORS = {"A": ACCENT, "B": GREEN, "C": ORANGE, "D": PURPLE}
GROUP_NAMES = {"A": "LAT 400+4000 Hz", "B": "VER 400+4000 Hz",
               "C": "LAT 200+4000 Hz", "D": "VER 200+4000 Hz"}

# Baerwald theoretical HTA curve (groove radius mm -> degrees)
_THEORY_R = np.array([57.5, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 146.5])
_THEORY_HTA = np.array([1.266490161, 1.172119584, 0.991819316, 0.822275392, 0.662927206,
    0.513253271, 0.372767823, 0.24101777, 0.117579957, 0.002058708, -0.1059164,
    -0.206692538, -0.300595442, -0.387931056, -0.468987029, -0.544034074, -0.613327202,
    -0.677106861, -0.735599962, -0.789020828, -0.837572057, -0.881445323, -0.920822102,
    -0.955874344, -0.986765098, -1.013649077, -1.036673188, -1.05597702, -1.071693294,
    -1.083948282, -1.092862193, -1.098549537, -1.101119451, -1.100676018, -1.097318552,
    -1.091141869, -1.082236537, -1.07068911, -1.056582351, -1.039995432, -1.021004128,
    -0.999680996, -0.976095542, -0.950314379, -0.922401378, -0.892417798, -0.860422429,
    -0.826471701, -0.790619811, -0.752918823, -0.713418776, -0.672167777, -0.629212092,
    -0.584596235, -0.538363043, -0.490553759, -0.4412081, -0.390364328, -0.338059311,
    -0.284328589, -0.229206428, -0.172725877, -0.11491882, -0.055816025, 0.004552809,
    0.066159007, 0.128974878, 0.192973671, 0.258129538, 0.324417501, 0.391813412,
    0.460293924, 0.52983646, 0.600419179, 0.67202095, 0.744621328, 0.818200522,
    0.892739375, 0.968219338, 1.044622449, 1.121931309, 1.200129066, 1.279199391,
    1.359126461, 1.43989494, 1.521489964, 1.603897124, 1.687102447, 1.771092386,
    1.855853801, 1.898519815])


# ═════════════════════════════════════════════════════════════════════════════
#  CORE MODULE LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_core():
    """Import vta_analyzer_core.py (or the newest vta_analyzer*.py) from this folder."""
    path = HERE / "vta_analyzer_core.py"
    if not path.exists():
        cands = sorted(HERE.glob("vta_analyzer*.py"), key=os.path.getmtime, reverse=True)
        if not cands:
            return None, None
        path = cands[0]
    spec = importlib.util.spec_from_file_location("vta_analyzer_core", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path.name


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — TRACK DETECTION  (same logic as the desktop script)
# ═════════════════════════════════════════════════════════════════════════════

def detect_tracks(path, fs, n_frames, peak, win_s=0.5, hop_s=0.25,
                  min_track_s=10.0, progress=None):
    """Same detection as the desktop script, but streams the file from disk in
    ~1 minute batches instead of holding the whole recording in memory."""
    win, hop = int(fs * win_s), int(fs * hop_s)
    freqs = np.fft.rfftfreq(win, 1 / fs)
    hann = np.hanning(win)[:, None]
    m_car = (freqs > 3600) & (freqs < 4400)
    m_400 = (freqs > 370) & (freqs < 430)
    m_200 = (freqs > 175) & (freqs < 225)

    starts = list(range(0, n_frames - win, hop))
    segments = []
    BATCH = 240                                   # windows per disk read (~60 s)
    with sf.SoundFile(path) as f:
        for b0 in range(0, len(starts), BATCH):
            batch = starts[b0:b0 + BATCH]
            blk0, blk1 = batch[0], batch[-1] + win
            f.seek(blk0)
            blk = f.read(blk1 - blk0, dtype="float32", always_2d=True)
            if peak > 0:
                blk /= peak
            if progress is not None:
                progress(b0 / max(len(starts), 1))
            for start in batch:
                o = start - blk0
                sc = np.fft.rfft(blk[o:o + win, :2].astype(np.float64) * hann, axis=0)
                spec_avg = (np.abs(sc[:, 0]) + np.abs(sc[:, 1])) * (2 / win) / 2
                t_mid = (start + win // 2) / fs

                if float(np.max(spec_avg[m_car])) < 0.01:
                    continue
                amp_400 = float(np.max(spec_avg[m_400]))
                amp_200 = float(np.max(spec_avg[m_200]))
                if amp_400 > 0.01:
                    f_mod, mod_amp = 400, amp_400
                elif amp_200 > 0.003:
                    f_mod, mod_amp = 200, amp_200
                else:
                    continue

                mask_m = (freqs > f_mod - 20) & (freqs < f_mod + 20)
                idx = int(np.argmax(np.abs(sc[mask_m, 0])))
                ph_l = float(np.angle(sc[mask_m, 0][idx]))
                ph_r = float(np.angle(sc[mask_m, 1][idx]))
                lr = float(np.degrees((ph_l - ph_r + np.pi) % (2 * np.pi) - np.pi))
                segments.append((t_mid, f_mod, "LAT" if abs(lr) < 60 else "VER", mod_amp))
            del blk

    if not segments:
        return []

    tracks = []
    seg_times, seg_fmod, seg_type, seg_amps = [segments[0][0]], segments[0][1], segments[0][2], [segments[0][3]]

    def flush():
        dur = seg_times[-1] - seg_times[0]
        if dur >= min_track_s:
            tracks.append({"t_start": seg_times[0], "t_end": seg_times[-1],
                           "f_mod": seg_fmod, "mod_type": seg_type,
                           "mod_amp_mean": float(np.mean(seg_amps)), "duration": dur})

    for t, fm, mt, ma in segments[1:]:
        if fm == seg_fmod and mt == seg_type and t - seg_times[-1] < 1.5:
            seg_times.append(t); seg_amps.append(ma)
        else:
            flush()
            seg_times, seg_fmod, seg_type, seg_amps = [t], fm, mt, [ma]
    flush()
    return tracks


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CLASSIFY (group + dB band + Gust Table 2 lookup)
# ═════════════════════════════════════════════════════════════════════════════

def classify_tracks(tracks, theta_vert, theta_lat):
    classified, notes, track_idx = [], [], 0
    for grp, mod_type, is_lat, f_mod, n_bands in GROUP_DEFS:
        grp_tracks, i = [], track_idx
        while len(grp_tracks) < n_bands and i < len(tracks):
            tr = tracks[i]
            if tr["f_mod"] == f_mod and tr["mod_type"] == mod_type:
                grp_tracks.append((i, tr))
            i += 1
        if not grp_tracks:
            notes.append(f"No tracks found for group {grp} ({mod_type} {f_mod} Hz).")
            continue
        if len(grp_tracks) < n_bands:
            notes.append(f"Group {grp} ({mod_type} {f_mod} Hz): found {len(grp_tracks)} of "
                         f"{n_bands} expected tracks; band labels assigned by amplitude "
                         f"order starting at +6 dB, so check them.")
        track_idx = grp_tracks[-1][0] + 1
        by_amp = sorted(grp_tracks, key=lambda x: x[1]["mod_amp_mean"])
        for (_, tr), band in zip(by_amp, BAND_LABELS[:len(by_amp)]):
            key = (f_mod, is_lat, band)
            if key not in GUST_TABLE:
                notes.append(f"No Gust Table 2 entry for {key}.")
                continue
            vel, rad = GUST_TABLE[key]
            classified.append({**tr, "group": grp, "band": band, "velocity": vel,
                               "radius_m": rad, "radius_mm": rad * 1000.0,
                               "theta_r": theta_lat if is_lat else theta_vert,
                               "is_lateral": is_lat})
    classified.sort(key=lambda x: x["t_start"])
    return classified, notes


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — VTA/HTA PER TRACK
# ═════════════════════════════════════════════════════════════════════════════

def analyse_tracks(tracks, path, fs, core, margin_frac, ds_factor=1, progress=None):
    """Reads each track from disk, one channel at a time. If ds_factor > 1 the
    channel is decimated first (anti-aliased) to cut memory and run time."""
    fs_eff = fs / ds_factor
    results = []
    for i, tr in enumerate(tracks):
        if progress is not None:
            progress(i / len(tracks), f"Track {i + 1}/{len(tracks)}: Grp {tr['group']} "
                     f"{tr['mod_type']} {tr['f_mod']} Hz {tr['band']}")
        margin = (tr["t_end"] - tr["t_start"]) * margin_frac
        s0 = int((tr["t_start"] + margin) * fs)
        s1 = int((tr["t_end"] - margin) * fs)
        with sf.SoundFile(path) as f:
            f.seek(s0)
            blk = f.read(s1 - s0, dtype="float32", always_2d=True)

        ch_results = {}
        for ch_idx, ch in enumerate(["L", "R"]):
            audio = blk[:, ch_idx].astype(np.float64)
            if ds_factor > 1:
                audio = resample_poly(audio, 1, ds_factor)
            if len(audio) < fs_eff * 2:
                continue
            pk = np.max(np.abs(audio))
            if pk < 1e-6:
                continue
            audio = audio / pk
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    ch_results[ch] = core.analyse(
                        audio, fs_eff, channel=ch,
                        f_low=float(tr["f_mod"]), f_high=4000.0,
                        R=tr["radius_m"], V_low=tr["velocity"],
                        theta_r=tr["theta_r"],
                        modulation="lateral" if tr["is_lateral"] else "vertical",
                        verbose=False, _demo=False)
                except Exception as exc:                       # noqa: BLE001
                    st.warning(f"Track {i + 1} ch {ch} failed: {exc}")
            del audio
        del blk
        if ch_results:
            results.append({**tr, "ch_results": ch_results})
        gc.collect()
    if progress is not None:
        progress(1.0, "Analysis complete")
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT
# ═════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(MID)
    for sp in ax.spines.values():
        sp.set_color(GREY)
    ax.tick_params(colors=WHITE, labelsize=10)
    ax.set_xlabel(xlabel, fontsize=11, color=WHITE)
    ax.set_ylabel(ylabel, fontsize=11, color=WHITE)
    ax.set_title(title, color=WHITE, fontsize=12, pad=8)
    ax.grid(True, color=GREY, alpha=0.4, linewidth=0.7, linestyle="--")
    ax.set_xlim(57, 146)


def plot_results(results, wav_name, theta_vert, theta_lat, title_rev):
    fig = plt.figure(figsize=(12, 10), facecolor=DARK)
    gs = GridSpec(2, 1, figure=fig, hspace=0.35, left=0.09, right=0.78, top=0.89, bottom=0.08)
    ax_hta, ax_vta = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _style_ax(ax_hta, "HTA (Horizontal Tracking Angle) vs Groove Radius", "Groove radius (mm)", "HTA (°)")
    _style_ax(ax_vta, "VTA (Vertical Tracking Angle) vs Groove Radius", "Groove radius (mm)", "VTA (°)")

    for ax, ref in [(ax_hta, theta_lat), (ax_vta, theta_vert)]:
        ax.axhline(ref, color=AMBER, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axhspan(ref - 1, ref + 1, color=AMBER, alpha=0.07)
    ax_hta.plot(_THEORY_R, _THEORY_HTA, color=GREEN, linewidth=1.4, linestyle=":", alpha=0.9, zorder=2)

    present = set()
    for res in results:
        grp, color = res["group"], GROUP_COLORS[res["group"]]
        r, ax = res["radius_mm"], (ax_hta if res["is_lateral"] else ax_vta)
        present.add(grp)
        for ch, cr in res["ch_results"].items():
            ms = 9 if ch == "L" else 8
            ax.scatter(r, cr["vta_deg"], color=color, marker="o" if ch == "L" else "s",
                       s=ms ** 2, zorder=5, edgecolors=DARK, linewidths=0.8)
            if ch == "L":
                ax.annotate(res["band"], xy=(r, cr["vta_deg"]), xytext=(4, 4),
                            textcoords="offset points", color=color, fontsize=8, alpha=0.85)
        cr = res["ch_results"]
        if "L" in cr and "R" in cr:
            ax.plot([r, r], [cr["L"]["vta_deg"], cr["R"]["vta_deg"]],
                    color=color, alpha=0.35, linewidth=1.2)

    for grp in "ABCD":
        grp_res = sorted([r for r in results if r["group"] == grp],
                         key=lambda x: x["radius_mm"], reverse=True)
        if not grp_res:
            continue
        ax = ax_hta if grp_res[0]["is_lateral"] else ax_vta
        for ch in "LR":
            xs = [r["radius_mm"] for r in grp_res if ch in r["ch_results"]]
            ys = [r["ch_results"][ch]["vta_deg"] for r in grp_res if ch in r["ch_results"]]
            if len(xs) >= 2:
                ax.plot(xs, ys, color=GROUP_COLORS[grp], linewidth=1.4,
                        linestyle="-" if ch == "L" else ":", alpha=0.6, zorder=3)

    for ax, groups in [(ax_hta, "AC"), (ax_vta, "BD")]:
        handles = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_NAMES[g]) for g in groups if g in present]
        if ax is ax_hta:
            handles.append(mlines.Line2D([], [], color=GREEN, linestyle=":", linewidth=1.4, label="Baerwald theoretical"))
        handles += [
            mlines.Line2D([], [], color=WHITE, marker="o", linestyle="None", markersize=7, label="Ch L"),
            mlines.Line2D([], [], color=WHITE, marker="s", linestyle="None", markersize=7, label="Ch R"),
            mlines.Line2D([], [], color=AMBER, linestyle="--", linewidth=1.2, label="θR reference"),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  fontsize=9, framealpha=0.5, facecolor=MID, edgecolor=GREY, labelcolor=WHITE)

    fig.suptitle("CBS STR-112  Side B  —  VTA & HTA vs Groove Radius",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.952, f"WAV: {wav_name}   Groove radii from White & Gust (1982) Table 2   [{title_rev}]",
             ha="center", color=WHITE, fontsize=9)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  TABLES
# ═════════════════════════════════════════════════════════════════════════════

def tracks_table(tracks):
    return pd.DataFrame([{
        "#": i + 1, "Grp": t["group"], "Type": t["mod_type"], "f_mod (Hz)": t["f_mod"],
        "Band": t["band"], "V (m/s)": t["velocity"], "R (mm)": round(t["radius_mm"], 1),
        "t_start (s)": round(t["t_start"], 1), "t_end (s)": round(t["t_end"], 1),
    } for i, t in enumerate(tracks)])


def results_table(results):
    rows = []
    for i, res in enumerate(results):
        for ch, cr in res["ch_results"].items():
            rows.append({
                "#": i + 1, "Grp": res["group"], "Type": res["mod_type"],
                "f_mod (Hz)": res["f_mod"], "Band": res["band"],
                "R (mm)": round(res["radius_mm"], 1), "Ch": ch,
                "Angle (°)": round(cr["vta_deg"], 2),
                "Error vs θR (°)": round(cr["vta_error_deg"], 2),
                "F signed (Hz)": round(cr["F_signed_hz"], 2),
                "φ (°)": round(cr["phi_deg"], 1),
                "SNR (dB)": round(cr["snr_db"], 1),
                "Confidence": cr["confidence"],
                "IM (%)": round(cr["sidebands"]["imd_pct"], 2),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="STR-112 Side B · VTA/HTA vs radius", page_icon="💿", layout="wide")

core, core_name = load_core()

st.title("CBS STR-112 · Side B · VTA & HTA vs groove radius")
st.caption(f"Script: **{APP_SCRIPT}** · Revision: **{APP_REV}** · "
           f"Core: **{core_name or 'NOT FOUND'}**")

if core is None:
    st.error("vta_analyzer_core.py was not found next to app.py. Add it to the repository and redeploy.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    margin_frac = st.slider(
        "Track margin trimmed at each end", 0.0, 0.30, 0.05, 0.01,
        help="Fraction of every detected track removed at both ends before analysis. "
             "0.05 = analyse the middle 90 %. Increase if band transitions leak in.")
    theta_vert = st.number_input("θR vertical (°)", value=16.5, step=0.1, format="%.1f",
                                 help="Gust measured value for STR-112 (CLD 445-2).")
    theta_lat = st.number_input("θR lateral (°)", value=0.0, step=0.1, format="%.1f")
    rpm = st.number_input("Turntable speed (rpm)", value=33.333, step=0.001, format="%.3f")
    min_track_s = st.number_input("Minimum track length (s)", value=10.0, min_value=2.0, step=1.0)
    downsample = st.checkbox(
        "Decimate to ~48 kHz for analysis", value=True,
        help="Only applies to recordings at 88.2 kHz or higher. Halves memory use and run time. "
             "All signals of interest are below 5 kHz, so results change very little. "
             "Turn off to analyse at the full recorded rate (needs much more memory).")
    st.markdown("---")
    st.markdown("Side B order expected: **A** LAT 400 ×4, **B** VER 400 ×3, "
                "**C** LAT 200 ×4, **D** VER 200 ×3 (18 dB bands omitted). "
                "Record flat / bypass, not through RIAA.")

up = st.file_uploader("Side B recording (stereo WAV, whole side)", type=["wav", "flac", "aiff", "aif"])

if up is None:
    st.info("Upload a stereo recording of the whole of Side B to begin.")
    st.stop()

# ── Save the upload to disk; the audio is never held in RAM as a whole ──────
file_key = hashlib.md5(f"{up.name}|{up.size}|{min_track_s}".encode()).hexdigest()

if st.session_state.get("file_key") != file_key:
    old = st.session_state.pop("wav_path", None)
    if old and os.path.exists(old):
        os.unlink(old)
    for k in ("tracks_raw", "results", "fig_png"):
        st.session_state.pop(k, None)
    gc.collect()
    suffix = Path(up.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        up.seek(0)
        while True:
            chunk = up.read(8 * 1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = tmp.name
    info = sf.info(tmp_path)
    if info.channels < 2:
        os.unlink(tmp_path)
        st.error("The file must be stereo (two channels).")
        st.stop()
    peak = 0.0
    with st.spinner("Scanning level…"):
        with sf.SoundFile(tmp_path) as f:
            for blk in f.blocks(blocksize=1 << 20, dtype="float32", always_2d=True):
                peak = max(peak, float(np.max(np.abs(blk))))
    st.session_state.update(wav_path=tmp_path, fs=float(info.samplerate),
                            n_frames=int(info.frames), peak=peak, file_key=file_key)
    bar = st.progress(0.0, text="Detecting tracks…")
    st.session_state["tracks_raw"] = detect_tracks(
        tmp_path, float(info.samplerate), int(info.frames), peak, min_track_s=min_track_s,
        progress=lambda f_: bar.progress(min(f_, 1.0), text="Detecting tracks…"))
    bar.empty()

wav_path, fs = st.session_state["wav_path"], st.session_state["fs"]
n_frames = st.session_state["n_frames"]
tracks_raw = st.session_state["tracks_raw"]
ds_factor = max(1, int(round(fs / 48000.0))) if (downsample and fs >= 88200) else 1
st.write(f"**{up.name}** · {n_frames / fs:.1f} s · {fs:.0f} Hz · "
         f"{len(tracks_raw)} track segments detected"
         + (f" · analysed at {fs / ds_factor:.0f} Hz" if ds_factor > 1 else ""))

if not tracks_raw:
    st.error("No tracks detected. Check that this is a flat-recorded STR-112 Side B file.")
    st.stop()

tracks, notes = classify_tracks(tracks_raw, theta_vert, theta_lat)
expected = sum(g[4] for g in GROUP_DEFS)
if len(tracks_raw) != expected:
    st.warning(f"{len(tracks_raw)} segments detected, {expected} expected. "
               "Check the table below before trusting the labels.")
for n in notes:
    st.warning(n)

st.subheader("1 · Detected tracks")
st.dataframe(tracks_table(tracks), hide_index=True, width="stretch")

st.subheader("2 · Analysis")
if st.button("Run VTA / HTA analysis", type="primary"):
    core.TURNTABLE_RPM = float(rpm)       # used by core.compute_vta at call time
    bar = st.progress(0.0, text="Starting…")
    results = analyse_tracks(tracks, wav_path, fs, core, margin_frac, ds_factor,
                             progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    st.session_state["results"] = results
    fig = plot_results(results, up.name, theta_vert, theta_lat, f"{APP_SCRIPT} {APP_REV}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK)
    st.session_state["fig_png"] = buf.getvalue()
    plt.close("all")
    gc.collect()

results = st.session_state.get("results")
if results:
    st.image(st.session_state["fig_png"], width="stretch")
    df = results_table(results)
    st.dataframe(df, hide_index=True, width="stretch")
    base = Path(up.name).stem
    c1, c2 = st.columns(2)
    c1.download_button("Download plot (PNG)", st.session_state["fig_png"],
                       file_name=f"{base}_radius_plot.png", mime="image/png")
    c2.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}_radius_results.csv", mime="text/csv")
    low = df[df["Confidence"] == "LOW"]
    if len(low):
        st.caption(f"{len(low)} channel reading(s) have LOW confidence (SNR < 6 dB). "
                   "This is expected close to the reference angle, where the FM deviation is small by design.")
