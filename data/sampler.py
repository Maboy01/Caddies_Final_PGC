from __future__ import annotations

import pandas as pd

RANDOM_STATE = 42


def sample_balanced_rows(dataframe: pd.DataFrame, max_videos: int) -> pd.DataFrame:
    clubs = sorted(dataframe["club"].dropna().unique())
    per_club = max(1, max_videos // len(clubs))
    selected_parts = []
    selected_indexes: set[int] = set()

    for club in clubs:
        group = dataframe[dataframe["club"] == club]
        take = min(len(group), per_club)
        sampled = group.sample(n=take, random_state=RANDOM_STATE)
        selected_parts.append(sampled)
        selected_indexes.update(sampled.index.tolist())

    selected = pd.concat(selected_parts, axis=0)
    remaining = max_videos - len(selected)

    if remaining > 0:
        leftovers = dataframe.drop(index=list(selected_indexes))
        extra = leftovers.sample(
            n=min(remaining, len(leftovers)),
            random_state=RANDOM_STATE,
        )
        selected = pd.concat([selected, extra], axis=0)

    return selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)