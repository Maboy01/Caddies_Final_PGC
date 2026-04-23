"""
data/sampler.py — Muestreo balanceado de videos por clase.

Cuando se limita el número total de videos a cargar, esta función distribuye
el cupo de forma equitativa entre las clases disponibles ('wood' e 'iron'),
evitando que el dataset quede sesgado hacia la clase más numerosa.
"""
from __future__ import annotations

import pandas as pd

RANDOM_STATE = 42  # Semilla para que el muestreo sea reproducible


def sample_balanced_rows(dataframe: pd.DataFrame, max_videos: int) -> pd.DataFrame:
    """
    Selecciona hasta `max_videos` filas del DataFrame distribuyendo el cupo
    de manera equitativa entre todas las clases de la columna 'club'.

    Algoritmo:
        1. Divide max_videos entre el número de clases (per_club).
        2. Para cada clase, toma min(videos_disponibles, per_club) muestras.
        3. Si el total seleccionado es menor que max_videos (porque alguna
           clase tenía menos videos que per_club), rellena con muestras
           adicionales del resto del dataset hasta completar el cupo.
        4. Baraja el resultado final para evitar ordenamiento por clase.

    Args:
        dataframe   : DataFrame con al menos la columna 'club'
        max_videos  : Número máximo de filas a devolver

    Retorna un DataFrame de hasta max_videos filas, balanceado entre clases,
    con índice reseteado y orden aleatorio.
    """
    clubs = sorted(dataframe["club"].dropna().unique())  # Clases únicas ordenadas
    per_club = max(1, max_videos // len(clubs))          # Cupo por clase (mínimo 1)
    selected_parts = []
    selected_indexes: set[int] = set()

    # Primera pasada: muestrea per_club videos de cada clase
    for club in clubs:
        group = dataframe[dataframe["club"] == club]
        take = min(len(group), per_club)                  # No pedir más de lo disponible
        sampled = group.sample(n=take, random_state=RANDOM_STATE)
        selected_parts.append(sampled)
        selected_indexes.update(sampled.index.tolist())   # Registra los índices ya usados

    selected = pd.concat(selected_parts, axis=0)
    remaining = max_videos - len(selected)  # Videos que faltan para llegar al cupo

    # Segunda pasada: rellena el cupo con videos sobrantes si alguna clase quedó corta
    if remaining > 0:
        leftovers = dataframe.drop(index=list(selected_indexes))  # Videos no seleccionados aún
        extra = leftovers.sample(
            n=min(remaining, len(leftovers)),
            random_state=RANDOM_STATE,
        )
        selected = pd.concat([selected, extra], axis=0)

    # Baraja para que los batches no estén ordenados por clase
    return selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
