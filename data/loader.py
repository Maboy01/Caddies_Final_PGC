from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from data.video import load_video_sequence

RANDOM_STATE = 42


def parse_events(value: object) -> np.ndarray:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return np.array([], dtype=int)

    if not isinstance(parsed, list):
        return np.array([], dtype=int)

    events = np.array(parsed, dtype=float)
    events = events[np.isfinite(events)]
    return np.rint(events).astype(int)


def load_metadata(csv_path: Path, videos_dir: Path, max_videos: int) -> pd.DataFrame:
    print("Loading GolfDB metadata...")
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.startswith("Unnamed")]
    dataframe["video_path"] = dataframe["id"].apply(
        lambda video_id: videos_dir / f"{int(video_id)}.mp4"
    )
    dataframe = dataframe[dataframe["video_path"].apply(lambda path: path.exists())].copy()
    dataframe = dataframe[dataframe["club"].isin(["driver", "fairway", "iron", "hybrid"])].copy()
    dataframe["club"] = dataframe["club"].map({
        "driver": "wood",
        "fairway": "wood",
        "iron": "iron",
        "hybrid": "iron",
    })

    if max_videos > 0 and len(dataframe) > max_videos:
        from data.sampler import sample_balanced_rows
        dataframe = sample_balanced_rows(dataframe, max_videos)

    club_counts = {
        club: int(count)
        for club, count in dataframe["club"].value_counts().sort_index().items()
    }
    print(f"   Videos selected: {len(dataframe)}")
    print(f"   Clubs: {club_counts}")
    return dataframe.reset_index(drop=True)


def build_video_tensors(
    dataframe: pd.DataFrame,
    label_encoder: LabelEncoder,
    sequence_length: int,
    frame_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    targets = []
    total_rows = len(dataframe)

    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        print(f"   Loading video {position}/{total_rows}: {Path(row.video_path).name}")
        events = parse_events(row.events)
        sequence = load_video_sequence(
            Path(row.video_path),
            events,
            sequence_length,
            frame_size,
        )
        sequences.append(sequence)
        targets.append(label_encoder.transform([str(row.club)])[0])

    sequence_tensor = torch.from_numpy(np.stack(sequences))
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return sequence_tensor, target_tensor