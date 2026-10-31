"""
Autonomous Audio-Visual Montage Generator (main entry point).

Automatically discovers subfolders in the project root, interprets subfolder names
for media category and semantic content, analyzes the audio track for ambiance and
beat-synced cut points, deduplicates and scores pictures, analyzes videos to extract
their best highlight segments, plans a beat-locked montage, and renders the final
compilation movie in `output/compilation_movie.mp4`.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Auto-bootstrap into the local .venv if invoked with global system Python
if importlib.util.find_spec("librosa") is None:
    _venv_py = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve():
        print(f"[Bootstrap] Switching to project virtual environment: {_venv_py}")
        sys.exit(subprocess.call([str(_venv_py), *sys.argv]))

from src.audio_analyzer import AudioAnalysisResult, AudioSection, CutPointCandidate, analyze_audio_track
from src.folder_scanner import scan_workspace
from src.montage_planner import MontagePlan, create_montage_plan
from src.picture_analyzer import PictureAnalysis, analyze_pictures
from src.renderer import render_montage
from src.video_analyzer import VideoSegmentCandidate, analyze_videos


def _write_markdown_summary(
    summary_path: Path,
    folder_summary: Dict[str, dict],
    audio_res: AudioAnalysisResult,
    all_pics: List[PictureAnalysis],
    selected_pics: List[PictureAnalysis],
    all_vids: List[VideoSegmentCandidate],
    selected_vids: List[VideoSegmentCandidate],
    plan: MontagePlan,
    output_video_path: Path,
) -> None:
    """Write a comprehensive human-readable Markdown report of the autonomous analysis & montage."""
    duplicates = [p for p in all_pics if p.is_duplicate]
    redundant_vids = [v for v in all_vids if v.is_redundant]

    lines = [
        "# Autonomous Audio-Visual Montage Summary",
        "",
        f"- **Generated Compilation Video**: `{output_video_path.name}`",
        f"- **Original Audio Track**: `{Path(audio_res.file_path).name}` ({audio_res.total_duration:.2f}s, {audio_res.tempo_bpm:.1f} BPM)",
        f"- **Final Montage Duration**: **{plan.final_duration:.2f}s** ({plan.final_duration/60.0:.2f} min)",
        f"- **Audio Shortened**: `{plan.audio_was_shortened}` — {plan.shortening_reason}",
        f"- **Total Beat-Synced Cuts**: {plan.total_clips} ({plan.movie_clips_used} movie highlight segments, {plan.picture_clips_used} curated pictures)",
        "",
        "## 1. Discovered Subfolders & Semantic Interpretation",
        "| Subfolder | Inferred Type | Style Tag | Preferred Ambiance | Files Found | Description |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for fname, info in folder_summary.items():
        c = info["counts"]
        c_str = ", ".join(f"{v} {k}" for k, v in c.items() if v > 0)
        amb = ", ".join(info["preferred_ambiance"])
        lines.append(
            f"| `{fname}` | {info['inferred_base_type']} | `{info['style']}` | {amb} | {c_str} | {info['description']} |"
        )

    lines.extend([
        "",
        "## 2. Audio Track Ambiance & Structural Sections",
        "| Section | Time Range (s) | Duration (s) | Ambiance | Mean Energy | Harmonic Ratio | Brightness | Cut Pacing (Beats) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for sec in audio_res.sections:
        active_marker = "" if sec.start_time < plan.final_duration else " *(after cut)*"
        lines.append(
            f"| S{sec.index + 1}{active_marker} | {sec.start_time:.1f} – {sec.end_time:.1f} | {sec.duration:.1f} | "
            f"**{sec.ambiance}** | {sec.mean_energy:.2f} | {sec.harmonic_ratio:.2f} | {sec.brightness:.2f} | {sec.recommended_cut_beats} beats |"
        )

    lines.extend([
        "",
        "## 3. Picture Analysis & Duplicate Elimination",
        f"- **Total Pictures Scanned**: {len(all_pics)}",
        f"- **Duplicates / Burst Shots Eliminated**: {len(duplicates)}",
        f"- **Top Unique Pictures Selected for Pool**: {len(selected_pics)}",
        f"- **Pictures Used in Final Montage**: {plan.picture_clips_used}",
        "",
        "### Sample Detected Duplicate / Burst Pairs (Rejected -> Kept Winner)",
        "| Rejected Duplicate / Burst Shot | Kept Best Picture (Cluster Winner) | Rejected Score | Winner Score |",
        "| :--- | :--- | :--- | :--- |",
    ])

    winner_score_map = {p.file_name: p.overall_score for p in all_pics}
    for dup in duplicates[:25]:
        w_score = winner_score_map.get(dup.duplicate_of or "", 0.0)
        lines.append(
            f"| `{dup.file_name}` | `{dup.duplicate_of}` | {dup.overall_score:.1f} | **{w_score:.1f}** |"
        )
    if len(duplicates) > 25:
        lines.append(f"| *...and {len(duplicates) - 25} more duplicate/burst photos* | | | |")

    lines.extend([
        "",
        "## 4. Movie Analysis & Best-Segment Extraction",
        f"- **Total Movies Scanned**: {len(all_vids)}",
        f"- **Redundant / Back-to-Back Similar Takes Filtered**: {len(redundant_vids)}",
        f"- **Distinct Best Segments Available**: {len(selected_vids)}",
        f"- **Movie Segments Used in Final Montage**: {plan.movie_clips_used}",
        "",
        "| Subfolder | Video File | Full Duration | Best Extracted Window | Quality Score | Motion | Stability | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for v in sorted(all_vids, key=lambda x: x.overall_score, reverse=True):
        status = (
            f"Redundant with `{v.redundant_with}`"
            if v.is_redundant
            else ("Selected Pool" if v.selected_for_pool else "Filtered")
        )
        lines.append(
            f"| `{v.folder_name}` | `{v.file_name}` | {v.video_duration:.1f}s | "
            f"`{v.start_time:.1f}s – {v.end_time:.1f}s` | **{v.overall_score:.1f}** | "
            f"{v.motion_score:.2f} | {v.stability_score:.2f} | {status} |"
        )

    lines.extend([
        "",
        "## 5. Final Beat-Synced Montage Edit Decision List (EDL)",
        "| Slot | Timeline (s) | Dur (s) | Section Ambiance | Type | Source File | Source Window / Effect | Match Reason |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for item in plan.items:
        win_or_fx = (
            f"`{item.source_start:.1f}s – {item.source_end:.1f}s`"
            if item.media_type == "movie"
            else f"`{item.ken_burns_effect}`"
        )
        lines.append(
            f"| #{item.slot_index + 1} | {item.timeline_start:.2f} – {item.timeline_end:.2f} | "
            f"{item.duration:.2f} | `{item.section_ambiance}` | **{item.media_type}** | "
            f"`{item.folder_name}/{item.file_name}` | {win_or_fx} | {item.match_reason} |"
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Audio-Visual Montage Generator"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root workspace directory containing media subfolders (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/compilation_movie.mp4",
        help="Path for the generated compilation movie (default: output/compilation_movie.mp4)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run all audio, picture, and video analysis from scratch ignoring cache",
    )
    args = parser.parse_args()

    t_start = time.time()
    root_dir = Path(args.root).resolve()
    output_video_path = (root_dir / args.output).resolve()
    output_dir = output_video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "analysis_cache.json"

    print("=" * 76)
    print(" AUTONOMOUS AUDIO-VISUAL MONTAGE GENERATOR")
    print("=" * 76)

    # 1. Discover subfolders and media files
    workspace = scan_workspace(root_dir)
    if not workspace.audio_files:
        raise FileNotFoundError(
            f"No audio track found in any subfolder of '{root_dir}'."
        )
    if not workspace.picture_files and not workspace.movie_files:
        raise FileNotFoundError(
            f"No picture or movie files found in any subfolder of '{root_dir}'."
        )

    print("\n[Step 1/5] Discovered Subfolders & Semantic Roles:")
    for fname, meta in workspace.folder_summary.items():
        print(
            f"  - '{fname}' -> type={meta['inferred_base_type']}, "
            f"style={meta['style']}, files={meta['counts']} ({meta['description']})"
        )

    audio_media = workspace.audio_files[0]

    # Build fingerprint to validate cache
    fingerprint_items = [
        f"{mf.path.name}:{mf.path.stat().st_size}"
        for mf in (workspace.audio_files + workspace.picture_files + workspace.movie_files)
    ]
    fingerprint = "|".join(fingerprint_items)

    cached_data = None
    if cache_file.exists() and not args.no_cache:
        try:
            raw_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            if raw_cache.get("fingerprint") == fingerprint:
                cached_data = raw_cache
                print("\n[Cache] Loaded verified analysis cache from 'cache/analysis_cache.json'.")
        except Exception:
            cached_data = None

    # 2. Analyze Audio Track
    print("\n[Step 2/5] Audio Track Investigation (Ambiance, Beats, Sections)...")
    plot_path = output_dir / "audio_ambiance_timeline.png"
    audio_res = analyze_audio_track(audio_media.path, output_plot_path=plot_path)

    # 3. Analyze Pictures & Filter Duplicates
    print("\n[Step 3/5] Picture Quality Scoring & Duplicate/Burst Filtering...")
    if cached_data and "all_pics" in cached_data and "selected_pics" in cached_data:
        all_pics = [PictureAnalysis(**d) for d in cached_data["all_pics"]]
        selected_pics = [PictureAnalysis(**d) for d in cached_data["selected_pics"]]
        print(
            f"  [Cache] Restored {len(all_pics)} analyzed pictures "
            f"({len(selected_pics)} top deduplicated pictures)."
        )
    else:
        all_pics, selected_pics = analyze_pictures(workspace.picture_files)

    # 4. Analyze Movies & Extract Best Highlight Segments
    print("\n[Step 4/5] Movie Analysis & Best-Segment Extraction...")
    if cached_data and "all_vids" in cached_data and "selected_vids" in cached_data:
        all_vids = [VideoSegmentCandidate(**d) for d in cached_data["all_vids"]]
        selected_vids = [VideoSegmentCandidate(**d) for d in cached_data["selected_vids"]]
        print(
            f"  [Cache] Restored {len(all_vids)} analyzed videos "
            f"({len(selected_vids)} distinct best segments)."
        )
    else:
        all_vids, selected_vids = analyze_videos(workspace.movie_files)

    # Save analysis cache
    if cached_data is None:
        cache_payload = {
            "fingerprint": fingerprint,
            "all_pics": [p.to_dict() for p in all_pics],
            "selected_pics": [p.to_dict() for p in selected_pics],
            "all_vids": [v.to_dict() for v in all_vids],
            "selected_vids": [v.to_dict() for v in selected_vids],
        }
        cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")

    # 5. Plan & Render Compilation Montage
    print("\n[Step 5/5] Autonomous Montage Planning & Rendering...")
    plan = create_montage_plan(
        audio=audio_res,
        selected_pictures=selected_pics,
        selected_videos=selected_vids,
        crossfade_sec=0.33,
    )

    render_montage(plan, output_video_path)

    # Save JSON and Markdown reports
    report_json_path = output_dir / "montage_report.json"
    report_md_path = output_dir / "montage_summary.md"

    report_payload = {
        "folder_summary": workspace.folder_summary,
        "audio_analysis": {
            "file_path": audio_res.file_path,
            "total_duration": audio_res.total_duration,
            "tempo_bpm": audio_res.tempo_bpm,
            "num_beats": len(audio_res.beat_times),
            "sections": [s.__dict__ for s in audio_res.sections],
            "shortening_candidates": [c.__dict__ for c in audio_res.shortening_candidates],
        },
        "picture_summary": {
            "total_scanned": len(all_pics),
            "duplicates_found": sum(1 for p in all_pics if p.is_duplicate),
            "selected_pool_count": len(selected_pics),
            "pictures": [p.to_dict() for p in all_pics],
        },
        "video_summary": {
            "total_scanned": len(all_vids),
            "redundant_takes_found": sum(1 for v in all_vids if v.is_redundant),
            "selected_pool_count": len(selected_vids),
            "videos": [v.to_dict() for v in all_vids],
        },
        "montage_plan": plan.to_dict(),
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    _write_markdown_summary(
        summary_path=report_md_path,
        folder_summary=workspace.folder_summary,
        audio_res=audio_res,
        all_pics=all_pics,
        selected_pics=selected_pics,
        all_vids=all_vids,
        selected_vids=selected_vids,
        plan=plan,
        output_video_path=output_video_path,
    )

    elapsed = time.time() - t_start
    print("=" * 76)
    print(f" MONTAGE COMPLETE in {elapsed:.1f}s!")
    print(f" - Compilation Movie : {output_video_path}")
    print(f" - Montage Summary   : {report_md_path}")
    print(f" - Audio Timeline    : {plot_path}")
    print(f" - Detailed JSON     : {report_json_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()

