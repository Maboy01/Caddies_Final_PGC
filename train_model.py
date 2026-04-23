#!/usr/bin/env python3
"""
Script principal de entrenamiento del modelo CNN+LSTM para clasificar swings de golf.

Flujo general:
    1. Parsea argumentos de línea de comandos (épocas, batch size, etc.)
    2. Fija semillas aleatorias para reproducibilidad
    3. Carga los metadatos y videos del dataset GolfDB
    4. Entrena el modelo CNN+LSTM
    5. Genera gráficos y tablas de métricas
    6. Guarda el modelo entrenado en disco
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Usa backend sin ventana gráfica (necesario en servidores)
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from data.loader import load_metadata, build_video_tensors
from model.classifier import CnnLstmClassifier
from training.trainer import train_cnn_lstm
from training.metrics import (
    build_metrics,
    plot_history,
    plot_confusion_matrix,
    plot_class_metrics,
    plot_prediction_distribution,
    plot_confidence_distribution,
    save_training_tables,
)

warnings.filterwarnings("ignore")

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "GolfDB.csv"          # Metadatos del dataset
VIDEOS_DIR = BASE_DIR / "videos_160"        # Videos de swings (160px de alto)
OUTPUT_DIR = BASE_DIR / "model_results" / "cnn_lstm"  # Carpeta de salida
RANDOM_STATE = 42  # Semilla para reproducibilidad


def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos.

    Retorna un Namespace con:
        --max-videos      : Cuántos videos cargar (0 = todos)
        --sequence-length : Número de frames por video
        --frame-size      : Tamaño en píxeles de cada frame (cuadrado)
        --epochs          : Número máximo de épocas de entrenamiento
        --batch-size      : Tamaño del mini-batch
        --learning-rate   : Tasa de aprendizaje inicial
        --device          : Dispositivo de cómputo (auto/cpu/cuda)
        --output-dir      : Carpeta donde guardar resultados
    """
    parser = argparse.ArgumentParser(description="Train CNN+LSTM GolfDB model.")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def set_reproducibility() -> None:
    """
    Fija semillas aleatorias en Python, NumPy y PyTorch para que los
    experimentos sean reproducibles entre ejecuciones.
    """
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def resolve_device(requested: str) -> torch.device:
    """
    Selecciona el dispositivo de cómputo (GPU o CPU).

    Args:
        requested: "auto" detecta automáticamente, "cuda" fuerza GPU,
                   "cpu" fuerza CPU.

    Retorna el torch.device correspondiente.
    Lanza SystemExit si se pide CUDA y no está disponible.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_model(model, label_encoder, args, output_dir: Path) -> None:
    """
    Guarda el checkpoint del modelo entrenado en disco.

    El archivo .pt contiene:
        - model_state_dict : Pesos del modelo
        - classes          : Lista de clases (ej. ['iron', 'wood'])
        - sequence_length  : Longitud de secuencia usada en entrenamiento
        - frame_size       : Tamaño de frame usado en entrenamiento
        - model            : Nombre de la arquitectura ('cnn_lstm')
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": label_encoder.classes_.tolist(),
        "sequence_length": args.sequence_length,
        "frame_size": args.frame_size,
        "model": "cnn_lstm",
    }, output_dir / "cnn_lstm_model.pt")


def main() -> None:
    """
    Orquesta el pipeline completo de entrenamiento:
        1. Carga configuración y prepara carpetas de salida
        2. Carga metadatos del CSV y los videos como tensores
        3. Entrena el modelo CNN+LSTM
        4. Calcula métricas en el set de prueba
        5. Genera y guarda gráficos y tablas
        6. Persiste el modelo final
    """
    args = parse_args()
    set_reproducibility()
    args.output_dir.mkdir(parents=True, exist_ok=True)  # Crea carpeta de salida si no existe
    device = resolve_device(args.device)

    # Información de entorno para diagnóstico
    print("\nCNN+LSTM Golf Swing Training")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Max videos: {args.max_videos if args.max_videos else 'all'}\n")

    # --- Carga de datos ---
    dataframe = load_metadata(CSV_PATH, VIDEOS_DIR, args.max_videos)

    # Codifica las etiquetas de texto ('wood', 'iron') a enteros (0, 1)
    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe["club"].astype(str))

    # Convierte cada video a una secuencia de frames como tensor de PyTorch
    sequences, targets = build_video_tensors(
        dataframe, label_encoder, args.sequence_length, args.frame_size,
    )

    # --- Entrenamiento ---
    model, history, predictions, test_targets, confidences = train_cnn_lstm(
        sequences, targets,
        num_classes=len(label_encoder.classes_),
        args=args,
        device=device,
        output_dir=args.output_dir,
    )

    # --- Métricas y artefactos ---
    metrics = build_metrics(test_targets, predictions, confidences, label_encoder.classes_)

    print("Saving artifacts...")
    plot_history(history, args.output_dir)                   # Gráfico de pérdida/accuracy por época
    plot_confusion_matrix(metrics, args.output_dir)          # Matriz de confusión
    plot_class_metrics(metrics, args.output_dir)             # Precision/Recall/F1 por clase
    plot_prediction_distribution(metrics, args.output_dir)   # Distribución real vs predicha
    plot_confidence_distribution(metrics, args.output_dir)   # Histograma de confianzas
    save_training_tables(history, metrics, args.output_dir)  # Exporta CSVs con resultados
    save_model(model, label_encoder, args, args.output_dir)  # Guarda modelo final

    print("\nTraining completed.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
