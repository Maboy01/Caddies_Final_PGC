#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "GolfDB.csv"
VIDEOS_DIR = BASE_DIR / "videos_160"
OUTPUT_DIR = BASE_DIR / "model_results" / "cnn_lstm"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
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
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_model(model, label_encoder, args, output_dir: Path) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": label_encoder.classes_.tolist(),
        "sequence_length": args.sequence_length,
        "frame_size": args.frame_size,
        "model": "cnn_lstm",
    }, output_dir / "cnn_lstm_model.pt")


def main() -> None:
    args = parse_args()
    set_reproducibility()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print("\nCNN+LSTM Golf Swing Training")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Max videos: {args.max_videos if args.max_videos else 'all'}\n")

    dataframe = load_metadata(CSV_PATH, VIDEOS_DIR, args.max_videos)
    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe["club"].astype(str))
    sequences, targets = build_video_tensors(
        dataframe, label_encoder, args.sequence_length, args.frame_size,
    )

    model, history, predictions, test_targets, confidences = train_cnn_lstm(
        sequences, targets,
        num_classes=len(label_encoder.classes_),
        args=args,
        device=device,
        output_dir=args.output_dir,
    )
    metrics = build_metrics(test_targets, predictions, confidences, label_encoder.classes_)

    print("Saving artifacts...")
    plot_history(history, args.output_dir)
    plot_confusion_matrix(metrics, args.output_dir)
    plot_class_metrics(metrics, args.output_dir)
    plot_prediction_distribution(metrics, args.output_dir)
    plot_confidence_distribution(metrics, args.output_dir)
    save_training_tables(history, metrics, args.output_dir)
    save_model(model, label_encoder, args, args.output_dir)

    print("\nTraining completed.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()