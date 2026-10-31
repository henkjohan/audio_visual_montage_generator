"""
Autonomous Montage Director & Edit Decision List (EDL) Planner.

Matches the analyzed audio track's structural ambiance sections and beat grid
with the curated pool of deduplicated best pictures and best video segments.

Key responsibilities:
1. Determine optimal compilation duration: evaluates the high-quality visual pool
   and selects the best musical cadence / section boundary to shorten the audio
   track if a tighter compilation yields higher visual quality (never exceeding
   the audio track duration).
2. Generate beat-locked visual slots across each musical section according to
   the section's ambiance ('ambient', 'building', 'groovy', 'energetic', 'outro').
3. Assign the best-matching movie segment or picture to each slot using a
   multi-factor compatibility score (subfolder semantics, visual ambiance,
   quality score, motion/energy match, and visual variety).
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from src.audio_analyzer import AudioAnalysisResult, AudioSection
from src.picture_analyzer import PictureAnalysis
from src.video_analyzer import VideoSegmentCandidate


@dataclass
class MontageClipItem:
    slot_index: int
    timeline_start: float
    timeline_end: float
    duration: float
    section_index: int
    section_ambiance: str
    section_energy: float
    media_type: str  # 'movie' or 'picture'
    file_path: str
    file_name: str
    folder_name: str
    semantic_style: str
    orientation: str
    quality_score: float
    match_reason: str
    # For movie clips: source sub-clip window
    source_start: float = 0.0
    source_end: float = 0.0
    # For pictures: Ken Burns effect direction ('zoom_in', 'zoom_out', 'pan_left', 'pan_right')
    ken_burns_effect: str = "zoom_in"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MontagePlan:
    audio_file_path: str
    original_audio_duration: float
    final_duration: float
    audio_was_shortened: bool
    shortening_reason: str
    fade_in_duration: float
    fade_out_duration: float
    crossfade_duration: float
    tempo_bpm: float
    total_clips: int
    movie_clips_used: int
    picture_clips_used: int
    items: List[MontageClipItem] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _build_beat_slots_for_section(
    sec: AudioSection,
    beat_times: List[float],
    max_end_time: float,
) -> List[tuple]:
    """
    Partition a musical section [sec.start_time, min(sec.end_time, max_end_time)]
    into beat-aligned clip intervals.
    """
    t0 = sec.start_time
    t1 = min(sec.end_time, max_end_time)
    if t1 - t0 < 2.0:
        return []

    sec_beats = [b for b in beat_times if t0 - 0.05 <= b <= t1 + 0.05]
    # Use 6 to 8 beats per cut (~2.9s to 3.9s at ~123 BPM) so clips have breathing room
    step_beats = max(6, sec.recommended_cut_beats)

    if len(sec_beats) < step_beats + 1:
        return [(round(t0, 3), round(t1, 3))]

    slots = []
    cur_start = t0
    idx = step_beats
    while idx < len(sec_beats):
        cand_end = sec_beats[idx]
        # Ensure the remaining tail of the section isn't too short (< 2.5s)
        if t1 - cand_end < 2.5:
            slots.append((round(cur_start, 3), round(t1, 3)))
            cur_start = t1
            break
        slots.append((round(cur_start, 3), round(cand_end, 3)))
        cur_start = cand_end
        idx += step_beats

    if t1 - cur_start >= 2.0:
        slots.append((round(cur_start, 3), round(t1, 3)))

    return slots


def create_montage_plan(
    audio: AudioAnalysisResult,
    selected_pictures: List[PictureAnalysis],
    selected_videos: List[VideoSegmentCandidate],
    crossfade_sec: float = 0.33,
) -> MontagePlan:
    """
    Autonomously construct a beat-synced montage plan from the curated media pools.
    """
    print("[MontagePlanner] Designing autonomous beat-synced montage plan...", flush=True)

    # 1. Select elite subset of videos and pictures (do not dump everything in!)
    max_videos_to_use = min(len(selected_videos), max(10, int(round(len(selected_videos) * 0.85))))
    elite_videos = list(selected_videos[:max_videos_to_use])

    max_pics_to_use = min(len(selected_pictures), max(15, min(38, int(round(len(selected_pictures) * 0.75)))))
    elite_pictures = list(selected_pictures[:max_pics_to_use])

    total_elite_assets = len(elite_videos) + len(elite_pictures)
    # With 6-8 beats per clip (~3.2s average duration), calculate the natural visual duration
    ideal_visual_duration = total_elite_assets * 3.1

    # 2. Choose the best musical section boundary that fits our elite asset pool
    # without running out of clips mid-section and never exceeding audio.total_duration
    if ideal_visual_duration < audio.total_duration * 0.92 and audio.shortening_candidates:
        min_acceptable = max(90.0, ideal_visual_duration * 0.68)
        max_acceptable = min(audio.total_duration, ideal_visual_duration * 0.98)
        valid_cands = [
            c
            for c in audio.shortening_candidates
            if min_acceptable <= c.timestamp <= max_acceptable
        ]
        if not valid_cands:
            valid_cands = [
                c for c in audio.shortening_candidates if c.timestamp <= ideal_visual_duration
            ]
        if valid_cands:
            best_cand = max(
                valid_cands,
                key=lambda c: c.score - 0.20 * abs(c.timestamp - ideal_visual_duration * 0.88) / 60.0,
            )
            final_duration = min(audio.total_duration, best_cand.timestamp)
            audio_shortened = final_duration < audio.total_duration - 1.0
            shortening_reason = (
                f"Shortened audio from {audio.total_duration:.1f}s to {final_duration:.1f}s "
                f"at musical boundary ({best_cand.reason}) to keep pacing tight with only "
                f"the highest-scoring visual moments."
            )
        else:
            final_duration = min(audio.total_duration, round(ideal_visual_duration * 0.88, 2))
            audio_shortened = final_duration < audio.total_duration - 1.0
            shortening_reason = (
                f"Shortened audio to {final_duration:.1f}s to match curated top-quality media pool."
            )
    else:
        final_duration = audio.total_duration
        audio_shortened = False
        shortening_reason = "Used full audio track duration as curated media pool supports full length."

    # 3. Build beat-synced time slots up to final_duration
    raw_slots = []
    for sec in audio.sections:
        if sec.start_time >= final_duration - 1.0:
            break
        sec_slots = _build_beat_slots_for_section(sec, audio.beat_times, final_duration)
        for s_start, s_end in sec_slots:
            raw_slots.append((sec, s_start, s_end))

    # Ensure we have enough top-ranked pictures/videos in the active pool if raw_slots > total_elite_assets
    if len(raw_slots) > len(elite_videos) + len(elite_pictures):
        extra_needed = len(raw_slots) - (len(elite_videos) + len(elite_pictures))
        extra_pics = selected_pictures[len(elite_pictures) : len(elite_pictures) + extra_needed]
        elite_pictures.extend(extra_pics)

    total_slots = len(raw_slots)
    print(
        f"[MontagePlanner] Target duration: {final_duration:.2f}s across {total_slots} beat-synced cuts.",
        flush=True,
    )

    # 4. Assign the best matching visual asset to each slot
    unused_videos = list(elite_videos)
    unused_pics = list(elite_pictures)

    plan_items: List[MontageClipItem] = []
    kb_cycle = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
    kb_idx = 0

    consecutive_pics = 0
    consecutive_movies = 0
    last_folder = ""

    for slot_idx, (sec, s_start, s_end) in enumerate(raw_slots):
        slot_dur = round(s_end - s_start, 3)
        is_opening = slot_idx == 0
        is_closing = slot_idx == total_slots - 1

        want_movie_bonus = 0.0
        if is_opening or is_closing:
            want_movie_bonus = 25.0
        elif sec.ambiance in ("energetic", "building"):
            want_movie_bonus = 6.0
        elif sec.ambiance == "ambient":
            want_movie_bonus = 3.0

        remaining_slots = max(1, total_slots - slot_idx)
        target_video_ratio = len(unused_videos) / float(remaining_slots)
        if consecutive_pics >= 2 and unused_videos:
            want_movie_bonus += 20.0
        elif consecutive_movies >= 2 and unused_pics:
            want_movie_bonus -= 22.0
        elif target_video_ratio < 0.30 and not (is_opening or is_closing):
            want_movie_bonus -= 12.0

        best_vid: Optional[VideoSegmentCandidate] = None
        best_vid_score = -999.0
        best_vid_reason = ""

        for v in unused_videos:
            if v.video_duration < slot_dur + 0.4:
                continue
            score = v.overall_score + want_movie_bonus
            reasons = [f"quality={v.overall_score:.1f}"]

            if is_opening and v.semantic_style == "aerial":
                score += 20.0
                reasons.append("aerial opening establishing shot")
            elif is_closing and v.semantic_style == "aerial":
                score += 18.0
                reasons.append("aerial closing panoramic shot")
            elif sec.ambiance == "ambient" and v.semantic_style == "aerial":
                score += 14.0
                reasons.append("drone sweeping motion matches ambient section")
            elif sec.ambiance == "energetic" and v.semantic_style == "ground_camera":
                score += 14.0
                reasons.append("ground camera action matches high-energy section")
            elif sec.ambiance in v.visual_ambiance:
                score += 9.0
                reasons.append(f"matches '{sec.ambiance}' ambiance")

            energy_diff = abs(sec.mean_energy - v.motion_score)
            score += max(0.0, 8.0 * (1.0 - energy_diff))

            if v.folder_name != last_folder:
                score += 3.5

            if score > best_vid_score:
                best_vid_score = score
                best_vid = v
                best_vid_reason = ", ".join(reasons)

        best_pic: Optional[PictureAnalysis] = None
        best_pic_score = -999.0
        best_pic_reason = ""

        for p in unused_pics:
            score = p.overall_score
            reasons = [f"quality={p.overall_score:.1f}"]

            if sec.ambiance in p.visual_ambiance:
                score += 10.0
                reasons.append(f"matches '{sec.ambiance}' ambiance")
            if sec.ambiance in ("ambient", "building") and p.is_aerial_photo:
                score += 9.0
                reasons.append("aerial perspective photo")
            if sec.ambiance in ("groovy", "energetic") and p.has_faces:
                score += 7.0
                reasons.append("human subject presence")
            if p.orientation == "landscape":
                score += 3.0

            if score > best_pic_score:
                best_pic_score = score
                best_pic = p
                best_pic_reason = ", ".join(reasons)

        choose_video = False
        if best_vid is not None and best_pic is not None:
            choose_video = best_vid_score >= best_pic_score
        elif best_vid is not None:
            choose_video = True
        elif best_pic is not None:
            choose_video = False
        else:
            break

        if choose_video and best_vid is not None:
            unused_videos.remove(best_vid)
            consecutive_movies += 1
            consecutive_pics = 0
            last_folder = best_vid.folder_name

            mid_t = 0.5 * (best_vid.start_time + best_vid.end_time)
            needed_dur = slot_dur + crossfade_sec
            src_start = max(0.0, mid_t - 0.5 * needed_dur)
            src_end = src_start + needed_dur
            if src_end > best_vid.video_duration:
                src_end = best_vid.video_duration
                src_start = max(0.0, src_end - needed_dur)

            plan_items.append(
                MontageClipItem(
                    slot_index=slot_idx,
                    timeline_start=s_start,
                    timeline_end=s_end,
                    duration=slot_dur,
                    section_index=sec.index,
                    section_ambiance=sec.ambiance,
                    section_energy=sec.mean_energy,
                    media_type="movie",
                    file_path=best_vid.file_path,
                    file_name=best_vid.file_name,
                    folder_name=best_vid.folder_name,
                    semantic_style=best_vid.semantic_style,
                    orientation=best_vid.orientation,
                    quality_score=best_vid.overall_score,
                    match_reason=best_vid_reason,
                    source_start=round(src_start, 2),
                    source_end=round(src_end, 2),
                    ken_burns_effect="none",
                )
            )
        elif best_pic is not None:
            unused_pics.remove(best_pic)
            consecutive_pics += 1
            consecutive_movies = 0
            last_folder = best_pic.folder_name
            kb_effect = kb_cycle[kb_idx % len(kb_cycle)]
            kb_idx += 1

            plan_items.append(
                MontageClipItem(
                    slot_index=slot_idx,
                    timeline_start=s_start,
                    timeline_end=s_end,
                    duration=slot_dur,
                    section_index=sec.index,
                    section_ambiance=sec.ambiance,
                    section_energy=sec.mean_energy,
                    media_type="picture",
                    file_path=best_pic.file_path,
                    file_name=best_pic.file_name,
                    folder_name=best_pic.folder_name,
                    semantic_style=best_pic.semantic_style,
                    orientation=best_pic.orientation,
                    quality_score=best_pic.overall_score,
                    match_reason=best_pic_reason,
                    source_start=0.0,
                    source_end=round(slot_dur + crossfade_sec, 2),
                    ken_burns_effect=kb_effect,
                )
            )

    if plan_items:
        actual_end = plan_items[-1].timeline_end
        final_duration = min(final_duration, actual_end)

    movies_used = sum(1 for item in plan_items if item.media_type == "movie")
    pics_used = sum(1 for item in plan_items if item.media_type == "picture")

    print(
        f"[MontagePlanner] Planned {len(plan_items)} total clips "
        f"({movies_used} movie segments, {pics_used} pictures), "
        f"final duration={final_duration:.2f}s (original audio={audio.total_duration:.2f}s).",
        flush=True,
    )

    return MontagePlan(
        audio_file_path=audio.file_path,
        original_audio_duration=audio.total_duration,
        final_duration=round(final_duration, 2),
        audio_was_shortened=audio_shortened,
        shortening_reason=shortening_reason,
        fade_in_duration=1.5,
        fade_out_duration=3.5,
        crossfade_duration=crossfade_sec,
        tempo_bpm=audio.tempo_bpm,
        total_clips=len(plan_items),
        movie_clips_used=movies_used,
        picture_clips_used=pics_used,
        items=plan_items,
    )
