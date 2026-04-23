"""
data/video.py — Extracción y preprocesamiento de frames de video.

Convierte un archivo de video (.mp4) en un array numpy de frames
normalizados y redimensionados, listo para ser procesado por la CNN.

Los "eventos" son índices de frames clave del swing (marcados en el CSV),
que se usan como anclas para concentrar el muestreo en la parte relevante
del video (backswing → impacto → follow-through).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def select_evenly(values: np.ndarray, count: int) -> np.ndarray:
    """
    Selecciona exactamente `count` valores de `values` distribuidos de
    forma equiespaciada, preservando el orden.

    Útil para reducir un conjunto de índices candidatos a exactamente
    la longitud de secuencia requerida por el modelo.

    Args:
        values : Array de índices candidatos
        count  : Cantidad de elementos a seleccionar

    Retorna un subarray de tamaño min(len(values), count).
    """
    if len(values) == 0:
        return values.astype(int)
    if len(values) <= count:
        return values.astype(int)
    # Genera count posiciones linealmente espaciadas en el rango del array
    positions = np.linspace(0, len(values) - 1, count).astype(int)
    return values[positions].astype(int)


def build_frame_indexes(
    frame_count: int,
    sequence_length: int,
    events: np.ndarray,
) -> np.ndarray:
    """
    Calcula qué frames del video se deben extraer para construir la secuencia.

    Estrategia:
        - Si hay al menos 2 eventos válidos, recorta el video entre el
          primer y el último evento (la parte útil del swing).
        - Combina una rejilla uniforme de frames con los propios eventos
          como anclas (para no perder los momentos clave).
        - Selecciona exactamente `sequence_length` índices de esa combinación.
        - Hace padding con el último frame si faltan índices.

    Args:
        frame_count     : Total de frames en el video
        sequence_length : Número de frames a seleccionar
        events          : Índices de frames clave (pueden estar vacíos)

    Retorna un array de enteros de tamaño `sequence_length` con los índices
    de frames a extraer, recortados al rango válido [0, frame_count-1].
    """
    sequence_length = max(2, min(sequence_length, frame_count))  # Límites seguros

    # Solo considera eventos dentro del rango del video
    valid_events = events[(events >= 0) & (events < frame_count)]

    if len(valid_events) >= 2:
        # Recorta al segmento relevante del swing
        start_frame = int(valid_events[0])
        end_frame = int(valid_events[-1])
    else:
        # Sin eventos: usa el video completo
        start_frame = 0
        end_frame = frame_count - 1

    # Asegura que el rango sea válido
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = frame_count - 1

    # Crea una rejilla uniforme entre inicio y fin del swing
    timeline_indexes = np.linspace(start_frame, end_frame, sequence_length).astype(int)

    # Combina la rejilla con los eventos clave y elimina duplicados
    anchor_indexes = np.unique(np.concatenate([timeline_indexes, valid_events]))

    # Reduce al número exacto de frames requerido
    selected_indexes = select_evenly(anchor_indexes, sequence_length)

    # Si aún faltan frames (caso extremo), repite el último índice disponible
    if len(selected_indexes) < sequence_length:
        pad_count = sequence_length - len(selected_indexes)
        selected_indexes = np.pad(selected_indexes, (0, pad_count), mode="edge")

    # Garantiza que todos los índices estén dentro de los límites del video
    return np.clip(selected_indexes, 0, frame_count - 1).astype(int)


def load_video_sequence(
    video_path: Path,
    events: np.ndarray,
    sequence_length: int,
    frame_size: int,
) -> np.ndarray:
    """
    Abre un video y extrae una secuencia de frames preprocesados.

    Pasos por frame:
        1. Convierte BGR (formato OpenCV) a RGB
        2. Redimensiona a (frame_size × frame_size) con interpolación de área
        3. Normaliza valores de píxel de [0, 255] a [0.0, 1.0]
        4. Transpone de (H, W, C) a (C, H, W) — formato PyTorch

    Args:
        video_path      : Ruta al archivo de video (.mp4)
        events          : Índices de frames clave del swing
        sequence_length : Número de frames a extraer
        frame_size      : Tamaño del frame cuadrado de salida (en píxeles)

    Retorna un array numpy de forma (sequence_length, 3, frame_size, frame_size)
    con dtype float32. Si el video no se puede abrir, devuelve un array de ceros.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        # Video inaccesible: devuelve secuencia negra en lugar de fallar
        return np.zeros((sequence_length, 3, frame_size, frame_size), dtype=np.float32)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return np.zeros((sequence_length, 3, frame_size, frame_size), dtype=np.float32)

    # Determina qué frames extraer
    frame_indexes = build_frame_indexes(frame_count, sequence_length, events)
    frames = []
    empty_frame = np.zeros((3, frame_size, frame_size), dtype=np.float32)  # Frame de relleno

    for frame_index in frame_indexes:
        # Salta directamente al frame deseado (más eficiente que leer en secuencia)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()

        if not success or frame is None:
            frames.append(empty_frame)  # Frame corrupto: usa negro
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                           # BGR → RGB
        frame = cv2.resize(frame, (frame_size, frame_size),
                           interpolation=cv2.INTER_AREA)                          # Redimensiona
        frame = frame.astype(np.float32) / 255.0                                  # Normaliza [0,1]
        frame = np.transpose(frame, (2, 0, 1))                                    # HWC → CHW
        frames.append(frame)

    capture.release()
    return np.stack(frames).astype(np.float32)  # (sequence_length, 3, H, W)
