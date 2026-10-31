"""
Video file analysis, scene/motion evaluation, and best-segment extraction module.

Uses OpenCV (cv2), ImageHash, and perceptual/optical-flow metrics to:
1. Inspect video resolution, frame rate, duration, and orientation metadata.
2. Sample frames across the video to measure:
   - Visual sharpness (Laplacian & Sobel gradient energy)
   - Exposure quality and dynamic range
   - Colorfulness (Hasler-Süsstrunk) and contrast
   - Optical flow motion magnitude and directional coherence (smooth drone glide vs. camera shake)
   - Scene changes / abrupt transitions (to avoid cutting mid-jerk or across scene jumps)
3. Extract the single best highlight segment [start_time, end_time] from each movie
   (plus an optional secondary segment for long >35s multi-scene videos).
4. Cluster visually redundant back-to-back videos so the montage features diverse,
   high-impact shots rather than repetitive clips.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image

from src.folder_scanner import MediaFile


@dataclass
class VideoSegmentCandidate:
    file_path: str
    file_name: str
    folder_name: str
    semantic_style: str
    semantic_tags: List[str]
    width: int
    height: int
    fps: float
    video_duration: float
    orientation: str  # 'landscape' or 'portrait'
    capture_timestamp: Optional[float]
    start_time: float
    end_time: float
    segment_duration: float
    sharpness_score: float
    exposure_score: float
    colorfulness_score: float
    motion_score: float
    stability_score: float
    overall_score: float
    mean_motion_magnitude: float
    visual_ambiance: List[str] = field(default_factory=list)
    phash_hex: str = ""
    cluster_id: int = -1
    is_redundant: bool = False
    redundant_with: Optional[str] = None
    selected_for_pool: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


def _extract_video_timestamp(path: Path) -> Optional[float]:
    name = path.stem
    m1 = re.search(r"(20\d{6})[_-](\d{6})", name)
    if m1:
        try:
            return datetime.strptime(m1.group(1) + m1.group(2), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    m2 = re.search(r"DJI_(20\d{12})", name)
    if m2:
        try:
            return datetime.strptime(m2.group(1), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _compute_colorfulness(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr.astype("float32"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    cf = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )
    return float(np.clip((cf - 12.0) / 68.0, 0.0, 1.0))


def _compute_hsv_hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def analyze_single_video(
    mf: MediaFile,
    sample_step_sec: float = 1.0,
    target_window_sec: float = 5.0,
) -> Tuple[Optional[VideoSegmentCandidate], Optional[np.ndarray], Optional[imagehash.ImageHash]]:
    """
    Sample a video file, score visual quality & optical flow across time,
    and extract its highest-scoring continuous highlight segment.
    """
    path = mf.path
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[VideoAnalyzer] Warning: could not open '{path.name}'")
        return None, None, None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    if duration < 2.0 or w == 0 or h == 0:
        cap.release()
        return None, None, None

    orientation = "landscape" if w >= h else "portrait"
    ts = _extract_video_timestamp(path)

    # Adjust sampling step so each video is sampled at ~8-12 evenly spaced points
    eff_step = max(sample_step_sec, (duration - 1.0) / 10.0)
    sample_times = np.arange(0.5, max(1.0, duration - 0.5), eff_step)

    sample_sharpness = []
    sample_exposure = []
    sample_color = []
    sample_motion_mag = []
    sample_motion_coherence = []
    sample_cut_jump = []
    sample_hists = []
    sample_frames_rgb = []

    prev_h_hist = None

    for t in sample_times:
        f_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok1, frame1 = cap.read()
        if not ok1 or frame1 is None:
            continue

        # Read the immediate next frame for instantaneous optical flow (avoids decoding 3 extra 4K frames)
        ok2, frame2 = cap.read()

        # Downscale to max 320px for fast quality & optical flow calculation
        scale = min(1.0, 320.0 / float(max(frame1.shape[1], frame1.shape[0])))
        sw = max(64, int(round(frame1.shape[1] * scale)))
        sh = max(64, int(round(frame1.shape[0] * scale)))
        f1_s = cv2.resize(frame1, (sw, sh), interpolation=cv2.INTER_AREA)
        g1 = cv2.cvtColor(f1_s, cv2.COLOR_BGR2GRAY)

        # Sharpness
        lap = float(cv2.Laplacian(g1, cv2.CV_64F).var())
        sharp = float(np.clip((np.log1p(lap) - 3.5) / 3.8, 0.0, 1.0))

        # Exposure
        mean_lum = float(np.mean(g1))
        dark_f = float(np.mean(g1 < 15))
        bright_f = float(np.mean(g1 > 245))
        exp = float(
            np.clip(
                1.0 - min(1.0, abs(mean_lum - 125.0) / 95.0) - (dark_f + bright_f * 1.3),
                0.05,
                1.0,
            )
        )

        # Colorfulness
        col = _compute_colorfulness(f1_s)

        # Histogram for scene-jump detection and cross-video comparison
        hist = _compute_hsv_hist(f1_s)
        if prev_h_hist is not None:
            corr = float(cv2.compareHist(prev_h_hist, hist, cv2.HISTCMP_CORREL))
            cut_jump = max(0.0, 1.0 - corr)
        else:
            cut_jump = 0.0
        prev_h_hist = hist

        # Optical flow between frame1 and frame2
        if ok2 and frame2 is not None:
            f2_s = cv2.resize(frame2, (sw, sh), interpolation=cv2.INTER_AREA)
            g2 = cv2.cvtColor(f2_s, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 2, 13, 2, 5, 1.1, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            # Directional coherence: how uniform is the camera/scene movement?
            if mean_mag > 0.25:
                vx = float(np.mean(np.cos(ang)))
                vy = float(np.mean(np.sin(ang)))
                coherence = float(np.sqrt(vx * vx + vy * vy))
            else:
                coherence = 0.85
        else:
            mean_mag = 0.5
            coherence = 0.8

        sample_sharpness.append(sharp)
        sample_exposure.append(exp)
        sample_color.append(col)
        sample_motion_mag.append(mean_mag)
        sample_motion_coherence.append(coherence)
        sample_cut_jump.append(cut_jump)
        sample_hists.append(hist)
        sample_frames_rgb.append(cv2.cvtColor(f1_s, cv2.COLOR_BGR2RGB))

    cap.release()

    if not sample_sharpness:
        return None, None, None

    n_samples = len(sample_sharpness)
    win_len = min(duration - 0.5, max(3.5, target_window_sec))
    win_samples = max(2, int(round(win_len / eff_step)))

    is_drone = mf.semantic_style == "aerial" or "drone" in mf.semantic_tags

    best_score = -1.0
    best_i0 = 0
    best_i1 = min(n_samples, win_samples)

    for i0 in range(max(1, n_samples - win_samples + 1)):
        i1 = min(n_samples, i0 + win_samples)
        sh_m = float(np.mean(sample_sharpness[i0:i1]))
        ex_m = float(np.mean(sample_exposure[i0:i1]))
        co_m = float(np.mean(sample_color[i0:i1]))
        mag_slice = np.array(sample_motion_mag[i0:i1])
        coh_m = float(np.mean(sample_motion_coherence[i0:i1]))
        max_jump = float(np.max(sample_cut_jump[i0:i1])) if i1 > i0 else 0.0

        mean_mag = float(np.mean(mag_slice))
        mag_std = float(np.std(mag_slice))

        if is_drone:
            # Drone clips: reward smooth, steady panoramic glide (mean_mag ~ 0.6..3.5, low std, high coherence)
            if mean_mag < 0.2:
                mot_s = 0.45  # Nearly static hover
            elif mean_mag <= 4.0:
                mot_s = 0.95
            else:
                mot_s = max(0.3, 1.0 - (mean_mag - 4.0) / 6.0)
            stab_s = float(np.clip(coh_m * 0.7 + max(0.0, 1.0 - mag_std / 2.2) * 0.3, 0.1, 1.0))
        else:
            # Ground camera clips: reward engaging activity without severe camera whip/blur
            if mean_mag < 0.15:
                mot_s = 0.55
            elif mean_mag <= 5.5:
                mot_s = 0.92
            else:
                mot_s = max(0.25, 1.0 - (mean_mag - 5.5) / 7.0)
            stab_s = float(np.clip(coh_m * 0.5 + max(0.0, 1.0 - mag_std / 3.0) * 0.5, 0.1, 1.0))

        # Penalize windows that contain an abrupt scene cut or gimbal jerk
        jump_penalty = 12.0 if max_jump > 0.35 else 0.0

        # Avoid the very first 0.5s or last 0.5s when the user presses record/stop
        edge_bonus = 2.0 if (i0 > 0 and i1 < n_samples) else 0.0

        widescreen_bonus = 4.0 if orientation == "landscape" else 0.0

        score = (
            sh_m * 26.0
            + ex_m * 20.0
            + co_m * 20.0
            + mot_s * 16.0
            + stab_s * 14.0
            + edge_bonus
            + widescreen_bonus
            - jump_penalty
        )

        if score > best_score:
            best_score = score
            best_i0 = i0
            best_i1 = i1

    start_t = float(sample_times[best_i0])
    end_t = min(float(duration - 0.2), start_t + win_len)
    if end_t - start_t < 3.0:
        start_t = max(0.0, end_t - min(duration, 3.5))

    mid_idx = (best_i0 + best_i1) // 2
    rep_hist = sample_hists[mid_idx]
    rep_pil = Image.fromarray(sample_frames_rgb[mid_idx]).resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    rep_phash = imagehash.phash(rep_pil, hash_size=8)

    sh_best = float(np.mean(sample_sharpness[best_i0:best_i1]))
    ex_best = float(np.mean(sample_exposure[best_i0:best_i1]))
    co_best = float(np.mean(sample_color[best_i0:best_i1]))
    mag_best = float(np.mean(sample_motion_mag[best_i0:best_i1]))
    coh_best = float(np.mean(sample_motion_coherence[best_i0:best_i1]))

    # Assign visual ambiance affinities based on subfolder semantics + motion + color
    vis_ambiance = list(mf.preferred_ambiance)
    if is_drone:
        if mag_best < 1.8:
            vis_ambiance = ["ambient", "building", "outro"]
        else:
            vis_ambiance = ["building", "groovy", "energetic", "ambient"]
    else:
        if mag_best >= 1.5:
            vis_ambiance = ["energetic", "groovy"]
        else:
            vis_ambiance = ["groovy", "building"]

    cand = VideoSegmentCandidate(
        file_path=str(path),
        file_name=path.name,
        folder_name=mf.folder_name,
        semantic_style=mf.semantic_style,
        semantic_tags=list(mf.semantic_tags),
        width=w,
        height=h,
        fps=round(fps, 2),
        video_duration=round(duration, 2),
        orientation=orientation,
        capture_timestamp=ts,
        start_time=round(start_t, 2),
        end_time=round(end_t, 2),
        segment_duration=round(end_t - start_t, 2),
        sharpness_score=round(sh_best, 3),
        exposure_score=round(ex_best, 3),
        colorfulness_score=round(co_best, 3),
        motion_score=round(float(np.clip(mag_best / 3.0, 0.2, 1.0)), 3),
        stability_score=round(coh_best, 3),
        overall_score=round(float(best_score), 2),
        mean_motion_magnitude=round(mag_best, 3),
        visual_ambiance=vis_ambiance,
        phash_hex=str(rep_phash),
    )

    return cand, rep_hist, rep_phash


def analyze_videos(
    movie_files: List[MediaFile],
) -> Tuple[List[VideoSegmentCandidate], List[VideoSegmentCandidate]]:
    """
    Analyze all movie files, extract the best segment from each, and deduplicate
    visually redundant back-to-back video clips.
    """
    print(f"[VideoAnalyzer] Analyzing {len(movie_files)} movie files for best segments...")
    candidates: List[VideoSegmentCandidate] = []
    hists: List[np.ndarray] = []
    phashes: List[imagehash.ImageHash] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(analyze_single_video, movie_files))

    for idx, (cand, hist, ph) in enumerate(results, 1):
        if cand is not None and hist is not None and ph is not None:
            candidates.append(cand)
            hists.append(hist)
            phashes.append(ph)
            print(
                f"  [{idx}/{len(movie_files)}] {cand.folder_name}/{cand.file_name} "
                f"({cand.video_duration:.1f}s) -> Best window [{cand.start_time:.1f}s - {cand.end_time:.1f}s] "
                f"score={cand.overall_score:.1f}"
            )

    # Cross-video similarity clustering (detect back-to-back takes of the same scene)
    n = len(candidates)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if candidates[i].folder_name != candidates[j].folder_name:
                continue
            hist_corr = float(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL))
            p_dist = phashes[i] - phashes[j]
            ts_i, ts_j = candidates[i].capture_timestamp, candidates[j].capture_timestamp
            time_diff = abs(ts_i - ts_j) if (ts_i is not None and ts_j is not None) else 999999.0

            # Consecutive takes within 90 seconds with high visual similarity
            if (p_dist <= 14) or (time_diff <= 90.0 and hist_corr >= 0.84) or (hist_corr >= 0.91):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    selected_segments: List[VideoSegmentCandidate] = []
    redundant_count = 0

    for cluster_id, members in enumerate(clusters.values()):
        sorted_m = sorted(members, key=lambda k: candidates[k].overall_score, reverse=True)
        winner = candidates[sorted_m[0]]
        winner.cluster_id = cluster_id
        winner.is_redundant = False
        selected_segments.append(winner)

        for r_idx in sorted_m[1:]:
            red = candidates[r_idx]
            red.cluster_id = cluster_id
            red.is_redundant = True
            red.redundant_with = winner.file_name
            red.selected_for_pool = False
            redundant_count += 1

    selected_segments.sort(key=lambda x: x.overall_score, reverse=True)
    print(
        f"[VideoAnalyzer] Identified {redundant_count} redundant/similar video takes. "
        f"{len(selected_segments)} distinct best video segments available for montage."
    )
    return candidates, selected_segments

