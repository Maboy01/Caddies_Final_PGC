from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def select_evenly(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) == 0:
        return values.astype(int)
    if len(values) <= count:
        return values.astype(int)
    positions = np.linspace(0, len(values) - 1, count).astype(int)
    return values[positions].astype(int)


def build_frame_indexes(
    frame_count: int,
    sequence_length: int,
    events: np.ndarray,
) -> np.ndarray:
    sequence_length = max(2, min(sequence_length, frame_count))
    valid_events = events[(events >= 0) & (events < frame_count)]

    if len(valid_events) >= 2:
        start_frame = int(valid_events[0])
        end_frame = int(valid_events[-1])
    else:
        start_frame = 0
        end_frame = frame_count - 1

    if end_frame <= start_frame:
        start_frame = 0
        end_frame = frame_count - 1

    timeline_indexes = np.linspace(start_frame, end_frame, sequence_length).astype(int)
    anchor_indexes = np.unique(np.concatenate([timeline_indexes, valid_events]))
    selected_indexes = select_evenly(anchor_indexes, sequence_length)

    if len(selected_indexes) < sequence_length:
        pad_count = sequence_length - len(selected_indexes)
        selected_indexes = np.pad(selected_indexes, (0, pad_count), mode="edge")

    return np.clip(selected_indexes, 0, frame_count - 1).astype(int)


def load_video_sequence(
    video_path: Path,
    events: np.ndarray,
    sequence_length: int,
    frame_size: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return np.zeros((sequence_length, 3, frame_size, frame_size), dtype=np.float32)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return np.zeros((sequence_length, 3, frame_size, frame_size), dtype=np.float32)

    frame_indexes = build_frame_indexes(frame_count, sequence_length, events)
    frames = []
    empty_frame = np.zeros((3, frame_size, frame_size), dtype=np.float32)

    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()

        if not success or frame is None:
            frames.append(empty_frame)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    capture.release()
    return np.stack(frames).astype(np.float32)