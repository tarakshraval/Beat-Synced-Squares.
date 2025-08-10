#!/usr/bin/env python3
"""main.py — overlay beat-synced squares and connecting lines on a video.

Usage examples:
    # Process a video file
    python main.py -i sample_data/playing_dead.mp4 -o output.mp4

    # Run on live webcam feed (press 'q' to quit)
    python main.py -i 0
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import librosa
import moviepy.editor as mpy
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Helpers for Audio Processing
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str, sr: int = 22050) -> Path:
    """Write the audio track of a video to a temporary wav file."""
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    clip = mpy.VideoFileClip(video_path)
    if clip.audio:
        clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
        return wav_path
    raise ValueError("Video has no audio track.")


def _detect_onsets_from_file(wav_path: Path, sr: int = 22050) -> np.ndarray:
    """Return an array of onset times (in seconds) from an audio file."""
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


class MicBeatDetector(threading.Thread):
    """A thread that listens to the mic and detects beats, putting them in a queue."""

    def __init__(self, sample_rate=22050, chunk_duration=1.0):
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.onset_queue = deque()
        self.stop_event = threading.Event()

    def run(self):
        logging.info("Starting microphone beat detection...")
        with sd.InputStream(samplerate=self.sample_rate, channels=1) as stream:
            while not self.stop_event.is_set():
                audio_chunk, overflowed = stream.read(self.chunk_size)
                if overflowed:
                    logging.warning("Microphone input overflowed!")
                onsets = librosa.onset.onset_detect(
                    y=audio_chunk[:, 0], sr=self.sample_rate, units="time"
                )
                if len(onsets) > 0:
                    self.onset_queue.append(True)  # Signal that a beat occurred

    def has_beat(self) -> bool:
        """Check if a beat has been detected since the last check."""
        if self.onset_queue:
            self.onset_queue.popleft()
            return True
        return False

    def stop(self):
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Core visual classes and rendering logic
# ---------------------------------------------------------------------------

class TrackedPoint:
    """A feature point tracked across successive frames."""

    def __init__(
        self,
        pos: tuple[float, float],
        life: int,
        size: int,
        label: str,
        font_scale: float,
        text_color: tuple[int, int, int],
        vertical: bool,
    ):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size
        self.label = label
        self.font_scale = font_scale
        self.text_color = text_color
        self.vertical = vertical


def _draw_visuals(frame: np.ndarray, active_points: list[TrackedPoint], neighbor_links: int):
    """Draws squares, labels, and connecting lines onto a frame."""
    h, w, _ = frame.shape

    # Draw neighbor lines
    if neighbor_links > 0 and len(active_points) > 1:
        coords = [tp.pos for tp in active_points]
        for i, p in enumerate(coords):
            dists = [
                (j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if i != j
            ]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                pt1 = tuple(p.astype(int))
                pt2 = tuple(coords[j].astype(int))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw squares and labels
    for tp in active_points:
        x, y = tp.pos
        s = tp.size
        tl = (int(x - s / 2), int(y - s / 2))
        br = (int(x + s / 2), int(y + s / 2))

        # Clamp coordinates to be within frame boundaries
        tl_c = (max(0, tl[0]), max(0, tl[1]))
        br_c = (min(w, br[0]), min(h, br[1]))

        # Invert colors inside the box
        if tl_c[0] < br_c[0] and tl_c[1] < br_c[1]:
            roi = frame[tl_c[1] : br_c[1], tl_c[0] : br_c[0]]
            frame[tl_c[1] : br_c[1], tl_c[0] : br_c[0]] = 255 - roi

        cv2.rectangle(frame, tl, br, (255, 255, 255), 1)

        # Draw label
        if tp.vertical:
            y_cursor = tl[1] + int(14 * tp.font_scale)
            for char in tp.label:
                if y_cursor > br[1] - 5:
                    break
                cv2.putText(
                    frame, char, (tl[0] + 4, y_cursor),
                    cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA,
                )
                y_cursor += int(12 * tp.font_scale)
        else:
            cv2.putText(
                frame, tp.label, (tl[0] + 4, br[1] - 6),
                cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA,
            )
    return frame


def _spawn_points(
    n: int,
    w: int,
    h: int,
    active_points: list[TrackedPoint],
    min_size: int,
    max_size: int,
    bell_width: float,
    life_frames: int,
    kps: list[cv2.KeyPoint] | None = None,
):
    """Spawns n new TrackedPoints, either at keypoint locations or randomly."""
    spawned = 0
    if kps:  # Spawn at provided keypoints
        kps = sorted(kps, key=lambda k: k.response, reverse=True)
        for kp in kps:
            if spawned >= n:
                break
            x, y = kp.pt
            # Don't spawn too close to existing points
            if any(np.linalg.norm(tp.pos - (x, y)) < 20 for tp in active_points):
                continue
            spawned += 1
            size = int(np.clip(np.random.normal((min_size + max_size) / 2, (max_size - min_size) / bell_width), min_size, max_size))
            label = str(uuid.uuid4())[:6]
            font_scale = random.uniform(1.0, 1.8)
            text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])
            vertical = random.random() < 0.25
            active_points.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))
    else:  # Spawn randomly
        for _ in range(n):
            x, y = random.uniform(0, w), random.uniform(0, h)
            size = int(np.clip(np.random.normal((min_size + max_size) / 2, (max_size - min_size) / bell_width), min_size, max_size))
            label = str(uuid.uuid4())[:6]
            font_scale = random.uniform(1.0, 1.8)
            text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])
            vertical = random.random() < 0.25
            active_points.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))


def _update_tracked_points(
    active: list[TrackedPoint], gray: np.ndarray, prev_gray: np.ndarray, w: int, h: int, jitter_px: float
) -> list[TrackedPoint]:
    """Tracks points using Optical Flow and prunes dead ones."""
    if not active:
        return []

    prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    new_active: list[TrackedPoint] = []
    for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
        if not ok:
            continue
        x, y = new_pt
        if 0 <= x < w and 0 <= y < h and tp.life > 0:
            tp.pos = new_pt
            tp.life -= 1
            if jitter_px > 0:
                tp.pos += np.random.normal(0, jitter_px, size=2)
                tp.pos = np.clip(tp.pos, [0, 0], [w - 1, h - 1])
            new_active.append(tp)
    return new_active


# ---------------------------------------------------------------------------
# Main Rendering Functions (File vs. Webcam)
# ---------------------------------------------------------------------------

def render_effect_to_file(args):
    """Processes a video file and saves the output."""
    logging.info(f"Processing video file: {args.input}")
    clip = mpy.VideoFileClip(args.input)
    fps = args.fps or clip.fps

    try:
        wav_path = _extract_audio(args.input)
        onset_times = _detect_onsets_from_file(wav_path)
        logging.info(f"{len(onset_times)} onsets detected in audio.")
    except (ValueError, IOError) as e:
        logging.warning(f"Could not process audio: {e}. No beat detection will occur.")
        onset_times = []

    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=args.orb_fast_threshold)
    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active
        frame = clip.get_frame(t)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # MoviePy is RGB, OpenCV is BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if prev_gray is not None:
            active = _update_tracked_points(active, gray, prev_gray, w, h, args.jitter_px)

        # Spawn on beats
        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            kps = orb.detect(gray, None)
            num_to_spawn = random.randint(1, args.pts_per_beat)
            _spawn_points(num_to_spawn, w, h, active, args.min_size, args.max_size, args.bell_width, args.life_frames, kps=kps)
            onset_idx += 1

        # Ambient spawns
        if args.ambient_rate > 0 and random.random() < args.ambient_rate / fps:
            _spawn_points(1, w, h, active, args.min_size, args.max_size, args.bell_width, args.life_frames)

        processed_frame = _draw_visuals(frame.copy(), active, args.neighbor_links)
        prev_gray = gray
        return cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) # Convert back to RGB for MoviePy

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration).set_fps(fps)
    if clip.audio:
        out_clip = out_clip.set_audio(clip.audio)

    logging.info(f"Writing output to {args.output}")
    out_clip.write_videofile(
        args.output, codec="libx264", audio_codec="aac", logger=None, verbose=False
    )
    logging.info("Done.")


def render_effect_on_webcam(args):
    """Runs the effect on a live webcam feed."""
    logging.info(f"Opening webcam: {args.input}")
    try:
        cam_index = int(args.input)
    except ValueError:
        logging.error(f"Invalid webcam index: {args.input}")
        return

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        logging.error(f"Cannot open webcam {cam_index}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    beat_detector = MicBeatDetector()
    beat_detector.start()

    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=args.orb_fast_threshold)
    active: list[TrackedPoint] = []
    prev_gray: np.ndarray | None = None

    logging.info("Webcam feed started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            active = _update_tracked_points(active, gray, prev_gray, w, h, args.jitter_px)

        # Spawn on beats from microphone
        if beat_detector.has_beat():
            logging.info("Beat detected!")
            kps = orb.detect(gray, None)
            num_to_spawn = random.randint(1, args.pts_per_beat)
            _spawn_points(num_to_spawn, w, h, active, args.min_size, args.max_size, args.bell_width, args.life_frames, kps=kps)

        # Ambient spawns
        if args.ambient_rate > 0 and random.random() < args.ambient_rate / fps:
            _spawn_points(1, w, h, active, args.min_size, args.max_size, args.bell_width, args.life_frames)

        processed_frame = _draw_visuals(frame.copy(), active, args.neighbor_links)
        cv2.imshow("Beat-Synced Squares", processed_frame)
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    beat_detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam feed stopped.")


# ---------------------------------------------------------------------------
# CLI and Main Execution
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Overlay beat-synced boxes + lines on a video.")
    p.add_argument("-i", "--input", type=str, default="0", help="Input video file or webcam index (e.g., '0')")
    p.add_argument("-o", "--output", type=str, default="output.mp4", help="Output video file (used only for file inputs)")
    p.add_argument("--fps", type=float, help="FPS for output (default: same as source)")
    p.add_argument("--life-frames", type=int, default=30, help="How many frames a point remains alive")
    p.add_argument("--pts-per-beat", type=int, default=15, help="Maximum new points to spawn on each beat")
    p.add_argument("--ambient-rate", type=float, default=2.0, help="Average number of random points spawned per second in silence")
    p.add_argument("--jitter-px", type=float, default=0.5, help="Per-frame positional jitter")
    p.add_argument("--min-size", type=int, default=20, help="Minimum square size")
    p.add_argument("--max-size", type=int, default=50, help="Maximum square size")
    p.add_argument("--neighbor-links", type=int, default=2, help="Number of neighbor edges per point")
    p.add_argument("--orb-fast-threshold", type=int, default=20, help="FAST threshold for ORB detector")
    p.add_argument("--bell-width", type=float, default=6.0, help="Divisor controlling bell curve width for size sampling")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Decide whether to process a file or run on webcam
    try:
        int(args.input)
        is_webcam = True
    except (ValueError, TypeError):
        is_webcam = False

    if is_webcam:
        render_effect_on_webcam(args)
    else:
        if not Path(args.input).is_file():
            logging.error(f"Input file not found: {args.input}")
            return
        render_effect_to_file(args)


if __name__ == "__main__":
    main()
