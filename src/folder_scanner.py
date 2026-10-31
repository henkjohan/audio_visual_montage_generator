"""
Subfolder discovery and semantic interpretation module.

Scans the project directory for media subfolders and interprets the folder names
to determine both the media type (audio, picture, movie) and the semantic content/style
(e.g., 'movies_drone' -> aerial/drone footage, 'movies' -> ground/general footage,
'pictures' -> still photography).
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
PICTURE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MOVIE_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

IGNORED_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "src",
    "output",
    "cache",
    "temp",
    "build",
    "dist",
}

# Semantic keyword mapping derived from subfolder names
SEMANTIC_PROFILES = {
    "drone": {
        "style": "aerial",
        "preferred_ambiance": ["ambient", "atmospheric", "building", "climax"],
        "motion_preference": "smooth_sweeping",
        "description": "Aerial / drone cinematography with sweeping panoramic views",
    },
    "aerial": {
        "style": "aerial",
        "preferred_ambiance": ["ambient", "atmospheric", "building", "climax"],
        "motion_preference": "smooth_sweeping",
        "description": "Aerial perspective footage",
    },
    "action": {
        "style": "action",
        "preferred_ambiance": ["energetic", "climax", "groovy"],
        "motion_preference": "dynamic",
        "description": "High-energy action footage",
    },
    "party": {
        "style": "social",
        "preferred_ambiance": ["energetic", "climax", "groovy"],
        "motion_preference": "dynamic",
        "description": "Social / party / club moments",
    },
    "sunset": {
        "style": "scenic_warm",
        "preferred_ambiance": ["ambient", "atmospheric", "groovy"],
        "motion_preference": "calm",
        "description": "Warm sunset / golden hour visuals",
    },
    "beach": {
        "style": "scenic_coastal",
        "preferred_ambiance": ["ambient", "groovy", "atmospheric"],
        "motion_preference": "calm",
        "description": "Coastal and beach scenes",
    },
    "landscape": {
        "style": "scenic",
        "preferred_ambiance": ["ambient", "atmospheric"],
        "motion_preference": "smooth_sweeping",
        "description": "Wide landscape shots",
    },
    "people": {
        "style": "social",
        "preferred_ambiance": ["groovy", "energetic"],
        "motion_preference": "moderate",
        "description": "Portraits and human moments",
    },
}


@dataclass
class MediaFile:
    path: Path
    folder_name: str
    media_type: str  # 'audio', 'picture', 'movie'
    semantic_tags: List[str] = field(default_factory=list)
    semantic_style: str = "general"
    preferred_ambiance: List[str] = field(default_factory=list)
    motion_preference: str = "moderate"


@dataclass
class WorkspaceMedia:
    audio_files: List[MediaFile] = field(default_factory=list)
    picture_files: List[MediaFile] = field(default_factory=list)
    movie_files: List[MediaFile] = field(default_factory=list)
    folder_summary: Dict[str, dict] = field(default_factory=dict)


def parse_folder_semantics(folder_name: str) -> dict:
    """
    Analyze a subfolder name (e.g., 'movies_drone', 'pictures', 'audio_track')
    to extract semantic tags, style, and preferred musical ambiance.
    """
    tokens = [t.lower() for t in re.split(r"[_\-\s]+", folder_name) if t]

    # Identify base category from folder tokens
    base_type = None
    for t in tokens:
        if t in {"audio", "music", "track", "tracks", "sound", "soundtrack", "song"}:
            base_type = "audio"
            break
        elif t in {"movie", "movies", "video", "videos", "clip", "clips", "footage"}:
            base_type = "movie"
            break
        elif t in {"picture", "pictures", "photo", "photos", "image", "images", "pic", "pics"}:
            base_type = "picture"
            break

    # Extract descriptor tokens (everything beyond the base media type word)
    descriptor_tokens = [
        t
        for t in tokens
        if t
        not in {
            "audio",
            "music",
            "track",
            "tracks",
            "sound",
            "movie",
            "movies",
            "video",
            "videos",
            "clip",
            "clips",
            "picture",
            "pictures",
            "photo",
            "photos",
            "image",
            "images",
        }
    ]

    style = "general"
    preferred_ambiance = ["groovy", "energetic", "ambient"]
    motion_preference = "moderate"
    description = f"General {base_type or 'media'} content from '{folder_name}'"

    if base_type == "movie" and not descriptor_tokens:
        # Plain 'movies' folder: ground-level / personal camera videos
        style = "ground_camera"
        preferred_ambiance = ["groovy", "energetic", "climax"]
        motion_preference = "dynamic_stable"
        description = "Ground-level / handheld camera videos (human perspective, lively moments)"
    elif base_type == "picture" and not descriptor_tokens:
        style = "photography"
        preferred_ambiance = ["ambient", "groovy", "building"]
        motion_preference = "still"
        description = "Still photography collection"

    for token in descriptor_tokens:
        if token in SEMANTIC_PROFILES:
            prof = SEMANTIC_PROFILES[token]
            style = prof["style"]
            preferred_ambiance = list(prof["preferred_ambiance"])
            motion_preference = prof["motion_preference"]
            description = prof["description"]
            break

    return {
        "folder_name": folder_name,
        "inferred_base_type": base_type,
        "semantic_tags": descriptor_tokens if descriptor_tokens else [style],
        "style": style,
        "preferred_ambiance": preferred_ambiance,
        "motion_preference": motion_preference,
        "description": description,
    }


def scan_workspace(root_dir: Path) -> WorkspaceMedia:
    """
    Scan all subfolders in `root_dir`, classify media files, and attach
    subfolder semantic metadata to each discovered file.
    """
    workspace = WorkspaceMedia()

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in IGNORED_DIRS or entry.name.startswith("."):
            continue

        semantics = parse_folder_semantics(entry.name)
        counts = {"audio": 0, "picture": 0, "movie": 0}

        for file_path in sorted(entry.rglob("*")):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()

            if ext in AUDIO_EXTENSIONS:
                mtype = "audio"
            elif ext in PICTURE_EXTENSIONS:
                mtype = "picture"
            elif ext in MOVIE_EXTENSIONS:
                mtype = "movie"
            else:
                continue

            media_item = MediaFile(
                path=file_path,
                folder_name=entry.name,
                media_type=mtype,
                semantic_tags=semantics["semantic_tags"],
                semantic_style=semantics["style"],
                preferred_ambiance=semantics["preferred_ambiance"],
                motion_preference=semantics["motion_preference"],
            )

            if mtype == "audio":
                workspace.audio_files.append(media_item)
            elif mtype == "picture":
                workspace.picture_files.append(media_item)
            elif mtype == "movie":
                workspace.movie_files.append(media_item)

            counts[mtype] += 1

        if sum(counts.values()) > 0:
            semantics["counts"] = counts
            workspace.folder_summary[entry.name] = semantics

    return workspace

