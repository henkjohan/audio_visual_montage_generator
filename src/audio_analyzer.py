"""
Audio track analysis module using librosa and scipy.

Analyzes an audio track for:
- Tempo (BPM), beat timestamps, and measure/phrase downbeats
- Onset strength and transient cut points
- Ambiance and mood features over time:
  - RMS energy (intensity / loudness contour)
  - Harmonic vs. Percussive energy ratio (HPSS: dreamy/ambient pads vs. rhythmic drums)
  - Spectral Centroid & Contrast (brightness vs. warm Balearic tone)
- Structural musical sections (Intro, Ambient/Breakdown, Groove, Building, Climax, Outro)
- Optimal musical cut/shortening points (cadences/phrase boundaries with natural energy dips)
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class AudioSection:
    index: int
    start_time: float
    end_time: float
    duration: float
    ambiance: str  # 'ambient', 'building', 'groovy', 'energetic', 'outro'
    mean_energy: float  # normalized 0..1
    harmonic_ratio: float  # 0..1 (higher = more melodic/ambient, lower = more percussive)
    brightness: float  # normalized 0..1 spectral centroid
    recommended_cut_beats: int  # how many beats per visual clip fits this section best
    preferred_sources: List[str] = field(default_factory=list)


@dataclass
class CutPointCandidate:
    timestamp: float
    score: float
    reason: str


@dataclass
class AudioAnalysisResult:
    file_path: str
    total_duration: float
    tempo_bpm: float
    beat_times: List[float]
    onset_times: List[float]
    sections: List[AudioSection]
    shortening_candidates: List[CutPointCandidate]
    timeline_times: List[float]
    timeline_energy: List[float]
    timeline_harmonic_ratio: List[float]
    timeline_brightness: List[float]

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


def _normalize(arr: np.ndarray) -> np.ndarray:
    p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
    if p95 - p5 < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - p5) / (p95 - p5), 0.0, 1.0)


def analyze_audio_track(
    audio_path: Path,
    output_plot_path: Optional[Path] = None,
    sr: int = 22050,
    hop_length: int = 512,
) -> AudioAnalysisResult:
    """
    Perform comprehensive musical, structural, and ambiance analysis on the audio track.
    """
    print(f"[AudioAnalyzer] Loading and analyzing '{audio_path.name}'...")
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    total_duration = float(len(y) / sr)

    # 1. Harmonic-Percussive Source Separation (HPSS)
    y_harm, y_perc = librosa.effects.hpss(y, margin=1.2)

    # 2. Tempo and Beat Tracking
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    tempo_bpm = float(np.ndim(tempo) and np.asarray(tempo).item() or tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    # Ensure beat grid covers from 0.0 to total_duration if intro is beatless/ambient
    beat_period = 60.0 / max(tempo_bpm, 60.0)
    if not beat_times or beat_times[0] > beat_period * 1.5:
        first_beat = beat_times[0] if beat_times else total_duration
        prepend = list(np.arange(0.0, first_beat - 0.2, beat_period))
        beat_times = [round(float(t), 3) for t in prepend] + beat_times
    if beat_times[-1] < total_duration - beat_period * 1.5:
        append = list(np.arange(beat_times[-1] + beat_period, total_duration, beat_period))
        beat_times = beat_times + [round(float(t), 3) for t in append]

    # 3. Prominent Onset Detection (for crisp transient accents)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        delta=0.15,
        wait=int(0.5 * sr / hop_length),
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length).tolist()

    # 4. Frame-level Ambiance Features
    rms_total = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_harm = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, hop_length=hop_length)[0]
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)

    frame_times = librosa.frames_to_time(
        np.arange(len(rms_total)), sr=sr, hop_length=hop_length
    )

    # Smooth curves (~1.5 second smoothing window) for macro-ambiance tracking
    smooth_sigma = max(1, int(1.5 * sr / hop_length))
    energy_norm = _normalize(gaussian_filter1d(rms_total, sigma=smooth_sigma))
    brightness_norm = _normalize(gaussian_filter1d(spec_centroid, sigma=smooth_sigma))

    harm_smooth = gaussian_filter1d(rms_harm, sigma=smooth_sigma)
    perc_smooth = gaussian_filter1d(rms_perc, sigma=smooth_sigma)
    harmonic_ratio = harm_smooth / (harm_smooth + perc_smooth + 1e-8)

    # 5. Structural Segmentation into Musical Sections
    # Target roughly 14-22 second sections (e.g., 14 to 18 sections over ~289s)
    n_sections = max(6, min(20, int(round(total_duration / 16.0))))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    seg_features = np.vstack([
        librosa.util.normalize(chroma, axis=1),
        librosa.util.normalize(mfcc, axis=1),
        energy_norm[np.newaxis, :] * 2.0,
    ])
    bound_frames = librosa.segment.agglomerative(seg_features, k=n_sections)
    bound_times_raw = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop_length)

    # Snap section boundaries to the nearest beat (preferably a 4-beat bar boundary)
    bar_times = beat_times[::4] if len(beat_times) >= 8 else beat_times
    snapped_bounds = [0.0]
    for bt in bound_times_raw[1:]:
        if bar_times:
            nearest_bar = min(bar_times, key=lambda b: abs(b - bt))
            if nearest_bar - snapped_bounds[-1] >= 8.0 and total_duration - nearest_bar >= 8.0:
                snapped_bounds.append(float(nearest_bar))
    snapped_bounds.append(round(total_duration, 3))

    sections: List[AudioSection] = []
    for idx in range(len(snapped_bounds) - 1):
        t0, t1 = snapped_bounds[idx], snapped_bounds[idx + 1]
        mask = (frame_times >= t0) & (frame_times <= t1)
        if not np.any(mask):
            continue

        sec_energy = float(np.mean(energy_norm[mask]))
        sec_harm = float(np.mean(harmonic_ratio[mask]))
        sec_bright = float(np.mean(brightness_norm[mask]))

        # Energy slope within section (is it building up or winding down?)
        e_slice = energy_norm[mask]
        if len(e_slice) > 10:
            half = len(e_slice) // 2
            slope = float(np.mean(e_slice[half:]) - np.mean(e_slice[:half]))
        else:
            slope = 0.0

        # Classify section ambiance and assign optimal visual pacing + subfolder affinities
        if idx == len(snapped_bounds) - 2 and sec_energy < 0.55:
            ambiance = "outro"
            rec_beats = 8
            pref_sources = ["drone", "photography", "scenic"]
        elif sec_energy < 0.42 or (sec_harm > 0.64 and sec_energy < 0.58):
            ambiance = "ambient"
            rec_beats = 8  # Longer, calmer holds (approx 3.8s - 4.2s at 120 BPM)
            pref_sources = ["drone", "aerial", "photography", "scenic"]
        elif slope > 0.14 and sec_energy < 0.75:
            ambiance = "building"
            rec_beats = 6  # Medium-paced progression (~2.8s - 3.2s)
            pref_sources = ["drone", "photography", "ground_camera"]
        elif sec_energy >= 0.68 and (sec_harm < 0.58 or sec_bright >= 0.55):
            ambiance = "energetic"
            rec_beats = 4  # Punchy 4-beat (1-bar) or 6-beat cuts (~2.0s - 3.0s)
            pref_sources = ["ground_camera", "action", "drone", "photography"]
        else:
            ambiance = "groovy"
            rec_beats = 6  # Flowing Balearic rhythm (~3.0s)
            pref_sources = ["photography", "ground_camera", "drone"]

        sections.append(
            AudioSection(
                index=idx,
                start_time=round(t0, 3),
                end_time=round(t1, 3),
                duration=round(t1 - t0, 3),
                ambiance=ambiance,
                mean_energy=round(sec_energy, 3),
                harmonic_ratio=round(sec_harm, 3),
                brightness=round(sec_bright, 3),
                recommended_cut_beats=rec_beats,
                preferred_sources=pref_sources,
            )
        )

    # 6. Identify Natural Audio Shortening / Fade-Out Candidates
    # Look for musical section boundaries and energy valleys between 45% and 95% of track
    shortening_candidates: List[CutPointCandidate] = []
    for sec in sections:
        t_end = sec.end_time
        if total_duration * 0.40 <= t_end <= total_duration * 0.96:
            # Check energy dip right around section boundary
            win_mask = (frame_times >= max(0, t_end - 2.5)) & (
                frame_times <= min(total_duration, t_end + 2.5)
            )
            local_e = float(np.mean(energy_norm[win_mask])) if np.any(win_mask) else 0.5
            # Lower local energy or transitioning into an ambient/outro section makes a cleaner fade-out
            cleanliness = 1.0 - local_e * 0.5
            if sec.ambiance in ("ambient", "outro"):
                cleanliness += 0.2
            shortening_candidates.append(
                CutPointCandidate(
                    timestamp=round(t_end, 3),
                    score=round(float(np.clip(cleanliness, 0.0, 1.0)), 3),
                    reason=f"End of section #{sec.index + 1} ({sec.ambiance}, local energy={local_e:.2f})",
                )
            )

    # Also include the full track end as a candidate
    shortening_candidates.append(
        CutPointCandidate(
            timestamp=round(total_duration, 3),
            score=0.85,
            reason="Full audio track natural ending",
        )
    )

    # Downsample timeline arrays for lightweight JSON serialization (~4 points/sec)
    step = max(1, int(0.25 * sr / hop_length))
    result = AudioAnalysisResult(
        file_path=str(audio_path),
        total_duration=round(total_duration, 3),
        tempo_bpm=round(tempo_bpm, 2),
        beat_times=[round(float(b), 3) for b in beat_times],
        onset_times=[round(float(o), 3) for o in onset_times],
        sections=sections,
        shortening_candidates=shortening_candidates,
        timeline_times=[round(float(t), 2) for t in frame_times[::step]],
        timeline_energy=[round(float(e), 3) for e in energy_norm[::step]],
        timeline_harmonic_ratio=[round(float(h), 3) for h in harmonic_ratio[::step]],
        timeline_brightness=[round(float(b), 3) for b in brightness_norm[::step]],
    )

    if output_plot_path is not None:
        _save_audio_plot(result, output_plot_path)

    print(
        f"[AudioAnalyzer] Done: {total_duration:.1f}s, Tempo={tempo_bpm:.1f} BPM, "
        f"{len(beat_times)} beats, {len(sections)} structural sections."
    )
    return result


def _save_audio_plot(result: AudioAnalysisResult, output_path: Path) -> None:
    """Save a diagnostic plot of audio energy, ambiance, and structural sections."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    times = np.array(result.timeline_times)
    energy = np.array(result.timeline_energy)
    harm = np.array(result.timeline_harmonic_ratio)
    bright = np.array(result.timeline_brightness)

    ambiance_colors = {
        "ambient": "#4FC3F7",
        "building": "#FFB74D",
        "groovy": "#81C784",
        "energetic": "#E57373",
        "outro": "#BA68C8",
    }

    ax0 = axes[0]
    ax0.plot(times, energy, color="#1E88E5", lw=1.8, label="Normalized RMS Energy")
    ax0.plot(times, bright, color="#FB8C00", lw=1.2, alpha=0.8, label="Spectral Brightness")
    for sec in result.sections:
        c = ambiance_colors.get(sec.ambiance, "#E0E0E0")
        ax0.axvspan(sec.start_time, sec.end_time, color=c, alpha=0.22)
        mid = 0.5 * (sec.start_time + sec.end_time)
        ax0.text(
            mid,
            1.03,
            f"S{sec.index+1}\n{sec.ambiance}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax0.set_ylim(0, 1.18)
    ax0.set_ylabel("Energy / Brightness")
    ax0.set_title(
        f"Audio Ambiance & Structural Analysis — {Path(result.file_path).name} "
        f"({result.tempo_bpm:.1f} BPM, {result.total_duration:.1f}s)"
    )
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    ax1.plot(
        times,
        harm,
        color="#8E24AA",
        lw=1.5,
        label="Harmonic Ratio (Higher = Ambient/Melodic, Lower = Percussive)",
    )
    for cand in result.shortening_candidates:
        ax1.axvline(cand.timestamp, color="#D32F2F", ls="--", lw=1.0, alpha=0.7)
    ax1.set_ylabel("Harmonic Ratio")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=140)
    plt.close(fig)

