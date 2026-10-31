# Autonomous Audio-Visual Montage Generator

An autonomous Python pipeline that analyzes an audio track alongside subfolders of pictures and videos, interprets subfolder names for semantic context, filters out duplicate/burst photos, extracts the best highlight segments from video files, and renders a beat-synced 1080p compilation movie (`output/compilation_movie.mp4`).

---

## Selected Python Packages & Roles

| Domain | Packages Used | Purpose in Pipeline |
| :--- | :--- | :--- |
| **Audio Analysis** | `librosa`, `scipy`, `soundfile`, `scikit-learn` | Extracts tempo (BPM), beat grid, transient onsets, Harmonic-Percussive Source Separation (`HPSS`), RMS energy curves, spectral centroid/brightness, agglomerative structural sections (`ambient`, `building`, `groovy`, `energetic`, `outro`), and natural musical shortening/fade-out points. |
| **Picture Analysis & Deduplication** | `Pillow`, `ImageHash` (`phash`, `dhash`), `opencv-python` | Applies EXIF orientation, extracts capture timestamps, computes perceptual hashes and 3D HSV color histograms to cluster duplicate/burst photos, and scores sharpness (Laplacian/Tenengrad), exposure balance, colorfulness (Hasler–Süsstrunk), contrast, and faces/saliency to select only the best picture per cluster. |
| **Video Analysis & Highlight Extraction** | `opencv-python`, `scenedetect`, `ImageHash` | Samples video frames to measure visual sharpness, exposure, colorfulness, scene jumps, and Farnebäck dense optical flow (camera motion magnitude & directional coherence). Extracts the single best sub-clip window (`[start_time, end_time]`) per movie and filters out visually redundant back-to-back takes. |
| **Montage Rendering** | `imageio-ffmpeg` (FFmpeg v7.1), `opencv-python`, `moviepy` | Renders a 1920x1080 @ 30fps H.264 + AAC compilation movie with sub-pixel Ken Burns zoom/pan animations on photos, ambient blurred-backdrop framing for portrait (`9:16` / `3:4`) media, beat-locked crossfades, and studio audio/video fade-in and fade-out. |

---

## Subfolder Naming Semantics

Simply place media files into subfolders in the project root. The scanner automatically infers both media type and semantic style from the subfolder name:

- `audio_track/` — Contains the audio track (`.mp3`, `.wav`, `.flac`, `.m4a`).
- `movies_drone/` — Inferred as `movie` with `aerial` / `drone` semantics (prioritized for establishing/closing shots, `ambient` harmonic passages, and sweeping panoramic reveals; evaluated for smooth, steady optical flow).
- `movies/` — Inferred as `movie` with `ground_camera` semantics (prioritized for `groovy` and `energetic` rhythmic sections; evaluated for dynamic activity and framing stability).
- `pictures/` — Inferred as still photography (deduplicated and matched to musical beats with smooth Ken Burns motion).
- Additional custom folders (e.g., `movies_action`, `pictures_sunset`, `pictures_beach`, `movies_party`) are automatically parsed by keyword and matched to corresponding musical sections.

---

## Running the Project

Using the project virtual environment (`.venv`):

```powershell
.\.venv\Scripts\python.exe main.py
```

### Generated Outputs (`output/`)

- `output/compilation_movie.mp4` — Final 1080p @ 30fps H.264/AAC compilation movie (never longer than the audio track, and automatically shortened to a natural musical cadence if a tighter duration better fits the curated media pool).
- `output/montage_summary.md` — Detailed human-readable report of subfolder semantics, audio sections, eliminated duplicate photos, selected video highlight windows, and the beat-synced Edit Decision List (EDL).
- `output/audio_ambiance_timeline.png` — Visual timeline plot of audio RMS energy, spectral brightness, harmonic ratio, and structural sections.
- `output/montage_report.json` — Complete machine-readable analysis and montage report.

