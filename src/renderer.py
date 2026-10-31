"""
High-performance 1080p Montage Video Renderer.

Renders the planned sequence of pictures and movie sub-clips to a 1920x1080 @ 30fps
H.264 MP4 file synchronized to the (optionally shortened and faded) audio track.

Features:
- Smooth sub-pixel Ken Burns effects (zoom_in, zoom_out, pan_left, pan_right) on pictures
- Automatic EXIF orientation handling for photos and rotation metadata for videos
- Ambient blurred-backdrop compositing for portrait (9:16 / 3:4) photos and videos
  so aspect ratios are never stretched or cropped destructively
- Fast FFmpeg input-seeking (-ss before -i) for instant 4K video sub-clip extraction
- Beat-locked crossfade transitions between consecutive clips with zero cumulative drift
- Deadlock-free single-pass FFmpeg pipe encoding with live per-clip progress output
"""

import subprocess
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

from src.montage_planner import MontageClipItem, MontagePlan


TARGET_W = 1920
TARGET_H = 1080
TARGET_FPS = 30


def _make_blurred_backdrop(bgr_img: np.ndarray, out_w: int = TARGET_W, out_h: int = TARGET_H) -> np.ndarray:
    """Create a softly blurred, darkened 16:9 ambient background from any BGR image."""
    h, w = bgr_img.shape[:2]
    target_aspect = out_w / float(out_h)
    src_aspect = w / float(h)

    if src_aspect > target_aspect:
        crop_w = int(round(h * target_aspect))
        x0 = (w - crop_w) // 2
        cropped = bgr_img[:, x0 : x0 + crop_w]
    else:
        crop_h = int(round(w / target_aspect))
        y0 = (h - crop_h) // 2
        cropped = bgr_img[y0 : y0 + crop_h, :]

    small = cv2.resize(cropped, (320, 180), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (31, 31), 0)
    darkened = (blurred.astype(np.float32) * 0.52).astype(np.uint8)
    return cv2.resize(darkened, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _prepare_picture_frames(
    item: MontageClipItem,
    num_frames: int,
    out_w: int = TARGET_W,
    out_h: int = TARGET_H,
) -> Iterator[np.ndarray]:
    """
    Generate `num_frames` BGR frames (1920x1080) for a still picture with a smooth
    Ken Burns zoom/pan animation.
    """
    with Image.open(item.file_path) as raw_im:
        im = ImageOps.exif_transpose(raw_im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        rgb = np.asarray(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    target_aspect = out_w / float(out_h)
    src_aspect = w / float(h)

    if w >= h:
        if src_aspect > target_aspect:
            crop_w = int(round(h * target_aspect))
            x0 = (w - crop_w) // 2
            cropped = bgr[:, x0 : x0 + crop_w]
        else:
            crop_h = int(round(w / target_aspect))
            y0 = (h - crop_h) // 2
            cropped = bgr[y0 : y0 + crop_h, :]

        margin_scale = 1.15
        overscan_w = int(round(out_w * margin_scale))
        overscan_h = int(round(out_h * margin_scale))
        overscan = cv2.resize(cropped, (overscan_w, overscan_h), interpolation=cv2.INTER_AREA)

        effect = item.ken_burns_effect
        for f_idx in range(num_frames):
            p = f_idx / float(max(1, num_frames - 1))
            p_smooth = 0.5 - 0.5 * float(np.cos(np.pi * p))

            if effect == "zoom_in":
                cur_scale = 1.0 + (margin_scale - 1.0) * (1.0 - p_smooth)
                cw = int(round(out_w * cur_scale))
                ch = int(round(out_h * cur_scale))
                x = (overscan_w - cw) // 2
                y = (overscan_h - ch) // 2
            elif effect == "zoom_out":
                cur_scale = 1.0 + (margin_scale - 1.0) * p_smooth
                cw = int(round(out_w * cur_scale))
                ch = int(round(out_h * cur_scale))
                x = (overscan_w - cw) // 2
                y = (overscan_h - ch) // 2
            elif effect == "pan_right":
                cw = int(round(out_w * 1.04))
                ch = int(round(out_h * 1.04))
                x = int(round((overscan_w - cw) * p_smooth))
                y = (overscan_h - ch) // 2
            else:  # pan_left
                cw = int(round(out_w * 1.04))
                ch = int(round(out_h * 1.04))
                x = int(round((overscan_w - cw) * (1.0 - p_smooth)))
                y = (overscan_h - ch) // 2

            sub = overscan[y : y + ch, x : x + cw]
            yield cv2.resize(sub, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        backdrop = _make_blurred_backdrop(bgr, out_w, out_h)
        margin_scale = 1.12
        fg_base_h = out_h
        fg_base_w = min(out_w - 40, int(round(fg_base_h * src_aspect)))
        over_h = int(round(fg_base_h * margin_scale))
        over_w = int(round(fg_base_w * margin_scale))
        fg_overscan = cv2.resize(bgr, (over_w, over_h), interpolation=cv2.INTER_AREA)

        for f_idx in range(num_frames):
            p = f_idx / float(max(1, num_frames - 1))
            p_smooth = 0.5 - 0.5 * float(np.cos(np.pi * p))
            if item.ken_burns_effect in ("zoom_in", "pan_right"):
                cur_scale = 1.0 + (margin_scale - 1.0) * (1.0 - p_smooth)
            else:
                cur_scale = 1.0 + (margin_scale - 1.0) * p_smooth

            cw = int(round(fg_base_w * cur_scale))
            ch = int(round(fg_base_h * cur_scale))
            x = (over_w - cw) // 2
            y = (over_h - ch) // 2
            fg_sub = cv2.resize(
                fg_overscan[y : y + ch, x : x + cw],
                (fg_base_w, fg_base_h),
                interpolation=cv2.INTER_LINEAR,
            )
            canvas = backdrop.copy()
            x_off = (out_w - fg_base_w) // 2
            if x_off >= 2:
                canvas[:, x_off - 2 : x_off] = 40
                canvas[:, x_off + fg_base_w : x_off + fg_base_w + 2] = 40
            canvas[:, x_off : x_off + fg_base_w] = fg_sub
            yield canvas


def _prepare_video_frames(
    item: MontageClipItem,
    num_frames: int,
    target_fps: int = TARGET_FPS,
    out_w: int = TARGET_W,
    out_h: int = TARGET_H,
) -> Iterator[np.ndarray]:
    """
    Fast FFmpeg input-seeking video segment decoder (`-ss` before `-i`).
    Avoids slow OpenCV 4K frame-by-frame seeking and scales in native C threads.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    read_dur = (num_frames / float(target_fps)) + 1.0

    if item.orientation == "landscape":
        vf = (
            f"fps={target_fps},"
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}"
        )
        dec_w, dec_h = out_w, out_h
    else:
        # Portrait video: decode at 608x1080 and composite onto blurred ambient backdrop
        dec_w = (int(round(out_h * 9.0 / 16.0)) // 2) * 2  # 608
        dec_h = out_h
        vf = (
            f"fps={target_fps},"
            f"scale={dec_w}:{dec_h}:force_original_aspect_ratio=increase,"
            f"crop={dec_w}:{dec_h}"
        )

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{item.source_start:.3f}",
        "-t",
        f"{read_dur:.3f}",
        "-i",
        item.file_path,
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    frame_bytes = dec_w * dec_h * 3
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    last_canvas: Optional[np.ndarray] = None

    try:
        for _ in range(num_frames):
            raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
            if len(raw) == frame_bytes:
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((dec_h, dec_w, 3))
                if item.orientation == "landscape":
                    last_canvas = arr
                else:
                    backdrop = _make_blurred_backdrop(arr, out_w, out_h)
                    x_off = (out_w - dec_w) // 2
                    if x_off >= 2:
                        backdrop[:, x_off - 2 : x_off] = 40
                        backdrop[:, x_off + dec_w : x_off + dec_w + 2] = 40
                    backdrop[:, x_off : x_off + dec_w] = arr
                    last_canvas = backdrop

            if last_canvas is not None:
                yield last_canvas
            else:
                yield np.zeros((out_h, out_w, 3), dtype=np.uint8)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()


def render_montage(
    plan: MontagePlan,
    output_video_path: Path,
    fps: int = TARGET_FPS,
    width: int = TARGET_W,
    height: int = TARGET_H,
) -> Path:
    """
    Render the complete montage plan to an MP4 file using FFmpeg.
    """
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_log_path = output_video_path.parent / "ffmpeg_encode.log"

    overlap_frames = max(1, int(round(plan.crossfade_duration * fps)))
    fade_in_frames = max(1, int(round(plan.fade_in_duration * fps)))
    fade_out_frames = max(1, int(round(plan.fade_out_duration * fps)))
    total_target_frames = int(round(plan.final_duration * fps))

    fade_out_start = max(0.0, plan.final_duration - plan.fade_out_duration)
    audio_filter = (
        f"afade=t=in:st=0:d={plan.fade_in_duration:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={plan.fade_out_duration:.2f}"
    )

    cmd = [
        ffmpeg_exe,
        "-y",
        "-nostats",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-ss",
        "0",
        "-t",
        f"{plan.final_duration:.3f}",
        "-i",
        plan.audio_file_path,
        "-af",
        audio_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_video_path),
    ]

    print(
        f"[Renderer] Starting 1080p@{fps}fps render of {len(plan.items)} clips "
        f"({plan.final_duration:.1f}s, {total_target_frames} frames) -> '{output_video_path.name}'",
        flush=True,
    )

    # Write FFmpeg stderr to a file instead of subprocess.PIPE to prevent OS pipe buffer deadlocks on Windows
    with open(ffmpeg_log_path, "wb") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_fh,
        )

        prev_tail: List[np.ndarray] = []
        emitted_frames = 0
        n_items = len(plan.items)

        try:
            for idx, item in enumerate(plan.items):
                is_last = idx == n_items - 1
                target_cum_end = int(round(item.timeline_end * fps))
                slot_main_frames = max(1, target_cum_end - emitted_frames)
                gen_frames_count = slot_main_frames + (0 if is_last else overlap_frames)

                if item.media_type == "picture":
                    frame_iter = _prepare_picture_frames(item, gen_frames_count, width, height)
                else:
                    frame_iter = _prepare_video_frames(item, gen_frames_count, fps, width, height)

                cur_tail: List[np.ndarray] = []

                for f_i, frame in enumerate(frame_iter):
                    if f_i < slot_main_frames:
                        if prev_tail and f_i < len(prev_tail):
                            alpha = (f_i + 1) / float(len(prev_tail) + 1)
                            out_frame = cv2.addWeighted(
                                prev_tail[f_i], 1.0 - alpha, frame, alpha, 0.0
                            )
                        else:
                            out_frame = frame

                        if emitted_frames < fade_in_frames:
                            v_alpha = emitted_frames / float(fade_in_frames)
                            out_frame = (out_frame.astype(np.float32) * v_alpha).astype(np.uint8)
                        elif total_target_frames - emitted_frames <= fade_out_frames:
                            rem = max(0, total_target_frames - emitted_frames)
                            v_alpha = rem / float(fade_out_frames)
                            out_frame = (out_frame.astype(np.float32) * v_alpha).astype(np.uint8)

                        assert proc.stdin is not None
                        proc.stdin.write(out_frame.tobytes())
                        emitted_frames += 1
                    else:
                        cur_tail.append(frame)

                prev_tail = cur_tail
                pct = 100.0 * emitted_frames / float(max(1, total_target_frames))
                print(
                    f"  [Renderer] [{idx + 1:02d}/{n_items:02d}] ({pct:5.1f}%) "
                    f"{item.media_type:7s} | {item.folder_name}/{item.file_name}",
                    flush=True,
                )

            if proc.stdin:
                proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                err_msg = ffmpeg_log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"FFmpeg exited with code {ret}:\n{err_msg}")

        except Exception:
            proc.kill()
            raise

    size_mb = output_video_path.stat().st_size / (1024.0 * 1024.0)
    print(f"[Renderer] Successfully generated '{output_video_path}' ({size_mb:.1f} MB)!", flush=True)
    return output_video_path
