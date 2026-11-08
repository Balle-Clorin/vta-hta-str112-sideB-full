# CBS STR-112 Side B: VTA & HTA vs groove radius

Streamlit app that analyses a recording of the whole of Side B of the CBS STR-112 test record.

It detects every test-tone track, labels it (Group A-D, +6 to +15 dB), takes peak velocity and groove radius from White & Gust (1982) Table 2, runs the FM-discriminator analysis (US Patent 4,359,768) on the L and R channel of each track, and plots HTA (lateral tracks) and VTA (vertical tracks) against groove radius.

| Group | Modulation | Tones | Bands |
|-------|------------|-------|-------|
| A | Lateral  | 400 + 4000 Hz | +6, +9, +12, +15 dB |
| B | Vertical | 400 + 4000 Hz | +6, +9, +12 dB |
| C | Lateral  | 200 + 4000 Hz | +6, +9, +12, +15 dB |
| D | Vertical | 200 + 4000 Hz | +6, +9, +12 dB |

The +18 dB bands are omitted.

## Files

- `app.py`: the Streamlit app (script name and revision are shown in the page header)
- `vta_analyzer_core.py`: single-track analyser, REV-23 or later. **Must sit next to `app.py`.**
- `requirements.txt`, `.streamlit/config.toml`, `LICENSE`

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Recording

Record flat / bypass, not through RIAA (RIAA adds about 50-60° of phase shift at 200-400 Hz and can flip the sign on one channel). Upload the whole side as one stereo WAV. The upload limit is set to 1000 MB in `.streamlit/config.toml`.

## Memory use (free hosting)

The recording is saved to disk and read in pieces; it is never loaded whole. For recordings at 88.2 kHz or higher the app decimates each track to about 48 kHz before analysis (sidebar option, on by default), which roughly halves memory. Streamlit itself keeps the uploaded file in RAM, so a smaller upload helps: FLAC, or a 48 kHz 24-bit WAV, are much lighter than a 96 kHz WAV and give the same measurement.

## Settings

- **Track margin**: fraction trimmed from each end of every track before analysis (0.05 default)
- **θR vertical / lateral**: reference recording angles (16.5° / 0°)
- **Turntable speed**: used for the formula coefficient (33.333 rpm)
- **Minimum track length**: tracks shorter than this are ignored
- **Decimate to ~48 kHz**: on by default for 88.2 kHz and above; turn off to analyse at the full rate

## Output

- Detected-track table (group, band, velocity, radius, start/end time)
- HTA and VTA vs radius plot (PNG download), with the Baerwald theoretical HTA curve
- Per-channel results table (CSV download): angle, error, signed F, φ, SNR, confidence, IM%

## Licence

PolyForm Noncommercial 1.0.0. See `LICENSE`.
