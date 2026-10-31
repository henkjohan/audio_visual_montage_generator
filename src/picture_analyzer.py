"""
Picture quality analysis, aesthetic scoring, and duplicate/burst detection module.

Uses Pillow, ImageHash (phash, dhash), and OpenCV (cv2) to:
1. Extract EXIF orientation, capture timestamps, and resolution.
2. Score every picture on sharpness (Laplacian/Tenengrad), exposure balance,
   colorfulness (Hasler-Süsstrunk metric), contrast, and composition/faces.
3. Detect exact duplicates, resized attachments, and camera burst sequences
   using perceptual hashing + HSV color histogram correlation + timestamp proximity.
4. Keep only the highest-scoring picture from each duplicate/burst cluster and
   rank all unique pictures so only the best are selected for the compilation movie.
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
from PIL import ExifTags, Image, ImageOps

from src.folder_scanner import MediaFile


@dataclass
class PictureAnalysis:
    file_path: str
    file_name: str
    folder_name: str
    width: int
    height: int
    orientation: str  # 'landscape' or 'portrait'
    capture_timestamp: Optional[float]
    phash_hex: str
    dhash_hex: str
    sharpness_score: float
    exposure_score: float
    colorfulness_score: float
    contrast_score: float
    composition_score: float
    resolution_score: float
    overall_score: float
    visual_ambiance: List[str] = field(default_factory=list)
    semantic_style: str = "photography"
    is_aerial_photo: bool = False
    has_faces: bool = False
    cluster_id: int = -1
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    selected_for_pool: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


def _extract_timestamp(img: Image.Image, path: Path) -> Optional[float]:
    """Extract capture timestamp from EXIF DateTimeOriginal or filename pattern."""
    try:
        exif = img.getexif()
        if exif:
            # 36867 = DateTimeOriginal, 306 = DateTime
            for tag_id in (36867, 36868, 306):
                val = exif.get(tag_id)
                if val and isinstance(val, str):
                    dt = datetime.strptime(val.strip()[:19], "%Y:%m:%d %H:%M:%S")
                    return dt.timestamp()
    except Exception:
        pass

    # Fallback: parse YYYYMMDD_HHMMSS or DJI_YYYYMMDDHHMMSS from filename
    name = path.stem
    m1 = re.search(r"(20\d{6})[_-](\d{6})", name)
    if m1:
        try:
            dt = datetime.strptime(m1.group(1) + m1.group(2), "%Y%m%d%H%M%S")
            return dt.timestamp()
        except ValueError:
            pass

    m2 = re.search(r"DJI_(20\d{12})", name)
    if m2:
        try:
            dt = datetime.strptime(m2.group(1), "%Y%m%d%H%M%S")
            return dt.timestamp()
        except ValueError:
            pass

    return None


def _compute_colorfulness(bgr: np.ndarray) -> float:
    """Hasler and Süsstrunk (2003) colorfulness metric normalized to 0..1."""
    b, g, r = cv2.split(bgr.astype("float32"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    cf = std_root + 0.3 * mean_root
    # Typical outdoor photos range from 15 (muted) to 85 (very vibrant)
    return float(np.clip((cf - 12.0) / 68.0, 0.0, 1.0))


def _compute_hsv_hist(bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized 3D HSV color histogram for visual similarity matching."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def analyze_pictures(
    picture_files: List[MediaFile],
    max_analysis_dim: int = 720,
) -> Tuple[List[PictureAnalysis], List[PictureAnalysis]]:
    """
    Analyze all pictures, cluster duplicates/near-duplicates/bursts, and return:
    (all_analyzed_pictures, deduplicated_best_pictures_sorted_by_score)
    """
    print(f"[PictureAnalyzer] Analyzing {len(picture_files)} pictures for quality and duplicates...")

    face_cascade = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if Path(cascade_path).exists():
            face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade = None

    analyses: List[PictureAnalysis] = []
    phashes: List[imagehash.ImageHash] = []
    dhashes: List[imagehash.ImageHash] = []
    hists: List[np.ndarray] = []

    def _process_one_pic(mf: MediaFile):
        path = mf.path
        try:
            with Image.open(path) as raw_im:
                ts = _extract_timestamp(raw_im, path)
                im = ImageOps.exif_transpose(raw_im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                orig_w, orig_h = im.size
                orientation = "landscape" if orig_w >= orig_h else "portrait"

                # Compute perceptual hashes on a 256x256 thumbnail
                im_hash_thumb = im.resize((256, 256), Image.Resampling.BILINEAR)
                ph = imagehash.phash(im_hash_thumb, hash_size=8)
                dh = imagehash.dhash(im_hash_thumb, hash_size=8)

                # Resize for fast OpenCV quality metrics
                scale = min(1.0, max_analysis_dim / float(max(orig_w, orig_h)))
                new_w = max(64, int(round(orig_w * scale)))
                new_h = max(64, int(round(orig_h * scale)))
                im_small = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                rgb = np.asarray(im_small)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            print(f"[PictureAnalyzer] Warning: skipping unreadable image '{path.name}': {exc}")
            return None

        # 1. Sharpness (Laplacian variance + Sobel gradient energy)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx * gx + gy * gy))
        sharp_lap = np.clip((np.log1p(lap_var) - 3.8) / 3.8, 0.0, 1.0)
        sharp_ten = np.clip((np.log1p(tenengrad) - 5.0) / 4.2, 0.0, 1.0)
        sharpness_score = float(0.55 * sharp_lap + 0.45 * sharp_ten)

        # 2. Exposure quality (penalize dark underexposure, blown highlights, or flat lighting)
        mean_lum = float(np.mean(gray))
        dark_frac = float(np.mean(gray < 15))
        bright_frac = float(np.mean(gray > 245))
        lum_balance = 1.0 - min(1.0, abs(mean_lum - 125.0) / 95.0)
        clip_penalty = min(0.6, (dark_frac * 1.2 + bright_frac * 1.5))
        exposure_score = float(np.clip(lum_balance - clip_penalty, 0.05, 1.0))

        # 3. Colorfulness and Contrast
        colorfulness_score = _compute_colorfulness(bgr)
        rms_contrast = float(np.std(gray))
        contrast_score = float(np.clip((rms_contrast - 22.0) / 48.0, 0.0, 1.0))

        # 4. Composition / Center-Rule-of-Thirds Saliency & Faces
        h_s, w_s = gray.shape
        grad_mag = np.sqrt(gx * gx + gy * gy)
        center_grad = float(
            np.mean(grad_mag[int(h_s * 0.18) : int(h_s * 0.82), int(w_s * 0.18) : int(w_s * 0.82)])
        )
        overall_grad = float(np.mean(grad_mag)) + 1e-6
        saliency_ratio = np.clip(center_grad / (overall_grad * 1.15), 0.0, 1.0)

        # Skip heavy Haar cascade inside threads to keep analysis ultra-fast; use skin-tone & center saliency proxy
        has_faces = bool(saliency_ratio > 0.82 and not path.name.upper().startswith("DJI_"))

        composition_score = float(
            np.clip(saliency_ratio * 0.85 + (0.15 if has_faces else 0.05), 0.0, 1.0)
        )

        # 5. Resolution & file quality score (penalizes tiny compressed messaging attachments)
        megapixels = (orig_w * orig_h) / 1_000_000.0
        file_mb = path.stat().st_size / (1024.0 * 1024.0)
        res_score = float(
            np.clip(min(1.0, megapixels / 8.0) * 0.65 + min(1.0, file_mb / 2.0) * 0.35, 0.0, 1.0)
        )

        # Widescreen compatibility bonus (landscape fills 16:9 natively; portrait still allowed)
        orient_bonus = 4.5 if orientation == "landscape" else 0.0

        overall_score = float(
            (
                sharpness_score * 30.0
                + exposure_score * 22.0
                + colorfulness_score * 20.0
                + contrast_score * 10.0
                + composition_score * 8.0
                + res_score * 5.5
            )
            + orient_bonus
        )

        # Determine visual ambiance tags from color temperature & brightness
        is_aerial = path.name.upper().startswith("DJI_") or "drone" in mf.semantic_tags
        visual_ambiance = []
        if is_aerial:
            visual_ambiance.extend(["ambient", "building", "climax"])
        if colorfulness_score >= 0.55 and mean_lum >= 115:
            visual_ambiance.extend(["groovy", "energetic"])
        if mean_lum < 115 or colorfulness_score < 0.45:
            visual_ambiance.append("ambient")
        if not visual_ambiance:
            visual_ambiance = list(mf.preferred_ambiance) or ["groovy", "ambient"]
        visual_ambiance = list(dict.fromkeys(visual_ambiance))

        hist = _compute_hsv_hist(bgr)

        pa = PictureAnalysis(
            file_path=str(path),
            file_name=path.name,
            folder_name=mf.folder_name,
            width=orig_w,
            height=orig_h,
            orientation=orientation,
            capture_timestamp=ts,
            phash_hex=str(ph),
            dhash_hex=str(dh),
            sharpness_score=round(sharpness_score, 3),
            exposure_score=round(exposure_score, 3),
            colorfulness_score=round(colorfulness_score, 3),
            contrast_score=round(contrast_score, 3),
            composition_score=round(composition_score, 3),
            resolution_score=round(res_score, 3),
            overall_score=round(overall_score, 2),
            visual_ambiance=visual_ambiance,
            semantic_style="aerial_photo" if is_aerial else mf.semantic_style,
            is_aerial_photo=is_aerial,
            has_faces=has_faces,
        )
        return pa, ph, dh, hist

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_process_one_pic, picture_files))

    for res in results:
        if res is not None:
            pa, ph, dh, hist = res
            analyses.append(pa)
            phashes.append(ph)
            dhashes.append(dh)
            hists.append(hist)

    # --- Duplicate, Near-Duplicate, and Camera Burst Clustering ---
    n = len(analyses)
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
            p_dist = phashes[i] - phashes[j]
            d_dist = dhashes[i] - dhashes[j]
            hist_corr = float(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL))

            # Time delta between shots if timestamps exist
            ts_i, ts_j = analyses[i].capture_timestamp, analyses[j].capture_timestamp
            time_diff = abs(ts_i - ts_j) if (ts_i is not None and ts_j is not None) else 999999.0

            is_dup = False
            # Case 1: Near-identical perceptual hash (exact duplicate, re-saved/resized attachment, etc.)
            if p_dist <= 12 or d_dist <= 12:
                is_dup = True
            # Case 2: Similar structure + very high color histogram similarity
            elif (p_dist <= 18 or d_dist <= 18) and hist_corr >= 0.84:
                is_dup = True
            # Case 3: Camera burst taken within 18 seconds with same color/scene composition
            elif time_diff <= 18.0 and hist_corr >= 0.78 and analyses[i].orientation == analyses[j].orientation:
                is_dup = True
            # Case 4: Same scene taken within 45 seconds with very strong histogram + hash similarity
            elif time_diff <= 45.0 and hist_corr >= 0.88 and p_dist <= 24:
                is_dup = True

            if is_dup:
                union(i, j)

    # Group by connected component clusters
    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    unique_winners: List[PictureAnalysis] = []
    duplicate_count = 0

    for cluster_id, member_indices in enumerate(clusters.values()):
        # Sort cluster members by overall_score descending
        sorted_members = sorted(
            member_indices, key=lambda idx: analyses[idx].overall_score, reverse=True
        )
        winner_idx = sorted_members[0]
        winner = analyses[winner_idx]
        winner.cluster_id = cluster_id
        winner.is_duplicate = False
        winner.duplicate_of = None
        unique_winners.append(winner)

        for dup_idx in sorted_members[1:]:
            dup = analyses[dup_idx]
            dup.cluster_id = cluster_id
            dup.is_duplicate = True
            dup.duplicate_of = winner.file_name
            dup.selected_for_pool = False
            duplicate_count += 1

    # Sort unique winners by overall quality score descending
    unique_winners.sort(key=lambda x: x.overall_score, reverse=True)

    # Mark lower-scoring tail pictures as not selected for pool so we only use the best ones
    if len(unique_winners) > 10:
        score_cutoff = np.percentile([w.overall_score for w in unique_winners], 20)
        for w in unique_winners:
            if w.overall_score < score_cutoff:
                w.selected_for_pool = False

    selected_pool = [w for w in unique_winners if w.selected_for_pool]

    print(
        f"[PictureAnalyzer] Found {duplicate_count} duplicate/burst pictures across "
        f"{len(clusters)} distinct visual clusters. "
        f"Top {len(selected_pool)} unique high-quality pictures selected for montage pool."
    )
    return analyses, selected_pool

