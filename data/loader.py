"""
data/loader.py — Carga y prepara los datos de entrenamiento del dataset GolfDB.

Responsabilidades:
    - Leer el CSV con metadatos de los swings (id, tipo de palo, eventos, etc.)
    - Filtrar los tipos de palo y agruparlos en dos clases: 'wood' e 'iron'
    - Balancear la cantidad de videos por clase si se limita el total
    - Convertir cada video a una secuencia de frames (tensor de PyTorch)
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from data.video import load_video_sequence

RANDOM_STATE = 42  # Semilla para muestreo reproducible


def parse_events(value: object) -> np.ndarray:
    """
    Convierte la columna 'events' del CSV (string con lista Python) a un
    array de enteros con los índices de frames clave del swing.

    Los eventos marcan posiciones temporales importantes dentro del video
    (ej. inicio del backswing, impacto, final del follow-through).

    Args:
        value: Valor crudo de la celda CSV, típicamente una cadena como "[10, 25, 40]".

    Retorna un array de enteros. Retorna array vacío si el valor es inválido.
    """
    try:
        parsed = ast.literal_eval(str(value))  # Parsea el string como estructura Python
    except (SyntaxError, ValueError):
        return np.array([], dtype=int)  # Valor malformado: devuelve vacío

    if not isinstance(parsed, list):
        return np.array([], dtype=int)  # No es lista: descarta

    events = np.array(parsed, dtype=float)
    events = events[np.isfinite(events)]       # Elimina NaN e Inf
    return np.rint(events).astype(int)         # Redondea al entero más cercano


def load_metadata(csv_path: Path, videos_dir: Path, max_videos: int) -> pd.DataFrame:
    """
    Lee el CSV de GolfDB, filtra los videos existentes y los clasifica en
    dos grupos de palos:
        - wood  : driver y fairway wood
        - iron  : iron y hybrid

    Si max_videos > 0 y hay más videos disponibles, aplica muestreo
    balanceado para no sobre-representar ninguna clase.

    Args:
        csv_path   : Ruta al archivo GolfDB.csv
        videos_dir : Carpeta que contiene los archivos .mp4
        max_videos : Límite de videos a usar (0 = sin límite)

    Retorna un DataFrame con columnas: id, club, events, video_path (y otras
    del CSV original), con índice reseteado.
    """
    print("Loading GolfDB metadata...")
    dataframe = pd.read_csv(csv_path)

    # Descarta columnas de índice extra que pandas puede generar al guardar CSV
    dataframe = dataframe.loc[:, ~dataframe.columns.str.startswith("Unnamed")]

    # Construye la ruta completa de cada video a partir de su id numérico
    dataframe["video_path"] = dataframe["id"].apply(
        lambda video_id: videos_dir / f"{int(video_id)}.mp4"
    )

    # Conserva solo los videos que realmente existen en disco
    dataframe = dataframe[dataframe["video_path"].apply(lambda path: path.exists())].copy()

    # Filtra palos soportados y fusiona en dos clases binarias
    dataframe = dataframe[dataframe["club"].isin(["driver", "fairway", "iron", "hybrid"])].copy()
    dataframe["club"] = dataframe["club"].map({
        "driver":  "wood",   # Palos de madera → clase 'wood'
        "fairway": "wood",
        "iron":    "iron",   # Hierros y híbridos → clase 'iron'
        "hybrid":  "iron",
    })

    # Aplica muestreo balanceado si se solicita un subconjunto
    if max_videos > 0 and len(dataframe) > max_videos:
        from data.sampler import sample_balanced_rows
        dataframe = sample_balanced_rows(dataframe, max_videos)

    # Muestra resumen de lo que se cargó
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
    """
    Convierte cada fila del DataFrame en una secuencia de frames lista para
    ser consumida por el modelo CNN+LSTM.

    Para cada video:
        1. Parsea los eventos (frames clave del swing)
        2. Extrae `sequence_length` frames distribuidos alrededor de esos eventos
        3. Redimensiona cada frame a (frame_size x frame_size) y normaliza a [0, 1]
        4. Codifica la etiqueta de clase ('wood'/'iron') como entero

    Args:
        dataframe       : DataFrame con columnas 'video_path', 'events' y 'club'
        label_encoder   : LabelEncoder ya entrenado con las clases disponibles
        sequence_length : Número de frames a extraer por video
        frame_size      : Dimensión (en píxeles) de cada frame cuadrado

    Retorna:
        sequences : Tensor de forma (N, sequence_length, 3, frame_size, frame_size)
        targets   : Tensor de forma (N,) con las clases codificadas como enteros
    """
    sequences = []
    targets = []
    total_rows = len(dataframe)

    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        print(f"   Loading video {position}/{total_rows}: {Path(row.video_path).name}")
        events = parse_events(row.events)          # Frames clave del swing
        sequence = load_video_sequence(
            Path(row.video_path),
            events,
            sequence_length,
            frame_size,
        )
        sequences.append(sequence)
        # Transforma la etiqueta de texto a su índice numérico
        targets.append(label_encoder.transform([str(row.club)])[0])

    # Apila todos los arrays y convierte a tensores de PyTorch
    sequence_tensor = torch.from_numpy(np.stack(sequences))
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return sequence_tensor, target_tensor
