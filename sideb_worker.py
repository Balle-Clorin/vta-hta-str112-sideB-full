"""
sideb_worker.py  —  runs the per-track VTA/HTA analysis in a short-lived process.

The analysis code retains memory after each track (and Python does not hand it
back to the OS), so running it inside the Streamlit server makes every run
bigger than the last. app.py therefore starts this script for a couple of
tracks at a time; when it exits, all of its memory is returned.

Usage (internal):  python sideb_worker.py job.json result.pkl
"""
import gc
import importlib.util
import json
import pickle
import sys
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_core(core_path):
    spec = importlib.util.spec_from_file_location("vta_analyzer_core", core_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def analyse_one(tr, path, fs, core, margin_frac, ds_factor):
    """Analyse L and R of one track. Returns (ch_results, error_messages)."""
    fs_eff = fs / ds_factor
    margin = (tr["t_end"] - tr["t_start"]) * margin_frac
    s0 = int((tr["t_start"] + margin) * fs)
    s1 = int((tr["t_end"] - margin) * fs)
    with sf.SoundFile(path) as f:
        f.seek(s0)
        blk = f.read(s1 - s0, dtype="float32", always_2d=True)
    ch_results, errors = {}, []
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
                errors.append(f"ch {ch} failed: {exc}")
        del audio
    del blk
    gc.collect()
    return ch_results, errors


def main():
    job_path, out_path = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    core = load_core(job["core_path"])
    core.TURNTABLE_RPM = float(job["rpm"])
    out = []
    for item in job["items"]:
        ch_results, errors = analyse_one(item["track"], job["wav_path"], job["fs"], core,
                                         job["margin_frac"], job["ds_factor"])
        out.append({"index": item["index"], "ch_results": ch_results, "errors": errors})
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
