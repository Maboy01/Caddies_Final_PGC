from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def build_metrics(
    test_targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, object]:
    labels = np.arange(len(class_names))
    accuracy = accuracy_score(test_targets, predictions)
    print(f"\nFinal test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_targets,
            predictions,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    precision, recall, f1_score, support = precision_recall_fscore_support(
        test_targets, predictions, labels=labels, zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "support": support,
        "test_targets": test_targets,
        "predictions": predictions,
        "confidences": confidences,
        "labels": labels,
        "class_names": class_names,
    }


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(14, 5))

    loss_axis.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#E63946")
    loss_axis.plot(epochs, history["validation_loss"], "s-", label="Validation Loss", color="#2E86AB")
    loss_axis.set_xlabel("Epoch", fontweight="bold")
    loss_axis.set_ylabel("Loss", fontweight="bold")
    loss_axis.set_title("CNN+LSTM Loss", fontweight="bold")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    accuracy_axis.plot(epochs, history["train_accuracy"], "o-", label="Train Accuracy", color="#06A77D")
    accuracy_axis.plot(epochs, history["validation_accuracy"], "s-", label="Validation Accuracy", color="#A23B72")
    accuracy_axis.set_xlabel("Epoch", fontweight="bold")
    accuracy_axis.set_ylabel("Accuracy", fontweight="bold")
    accuracy_axis.set_title("CNN+LSTM Accuracy", fontweight="bold")
    accuracy_axis.set_ylim([0, 1.05])
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(metrics: dict[str, object], output_dir: Path) -> None:
    matrix = confusion_matrix(
        metrics["test_targets"], metrics["predictions"], labels=metrics["labels"],
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=metrics["class_names"], yticklabels=metrics["class_names"],
        ax=axis, cbar_kws={"label": "Count"},
    )
    axis.set_xlabel("Prediction", fontsize=12, fontweight="bold")
    axis.set_ylabel("Actual", fontsize=12, fontweight="bold")
    axis.set_title("CNN+LSTM Confusion Matrix", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_class_metrics(metrics: dict[str, object], output_dir: Path) -> None:
    metric_values = [
        ("Precision", metrics["precision"], "#2E86AB"),
        ("Recall", metrics["recall"], "#A23B72"),
        ("F1-Score", metrics["f1"], "#F18F01"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (title, values, color) in zip(axes, metric_values):
        axis.bar(metrics["class_names"], values, color=color, alpha=0.75)
        axis.set_ylabel(title, fontweight="bold")
        axis.set_title(f"{title} by Class", fontweight="bold")
        axis.set_ylim([0, 1])
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(output_dir / "metrics_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_distribution(metrics: dict[str, object], output_dir: Path) -> None:
    labels = np.asarray(metrics["labels"])
    class_names = np.asarray(metrics["class_names"])
    test_targets = np.asarray(metrics["test_targets"])
    predictions = np.asarray(metrics["predictions"])
    actual_counts = np.array([(test_targets == label).sum() for label in labels])
    predicted_counts = np.array([(predictions == label).sum() for label in labels])
    x_positions = np.arange(len(class_names))
    bar_width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x_positions - bar_width / 2, actual_counts, width=bar_width, label="Actual", color="#2E86AB", alpha=0.8)
    axis.bar(x_positions + bar_width / 2, predicted_counts, width=bar_width, label="Predicted", color="#F18F01", alpha=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(class_names, rotation=35, ha="right")
    axis.set_ylabel("Samples", fontweight="bold")
    axis.set_title("Actual vs Predicted Distribution", fontweight="bold")
    axis.grid(True, alpha=0.3, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confidence_distribution(metrics: dict[str, object], output_dir: Path) -> None:
    confidences = np.asarray(metrics["confidences"], dtype=float)
    test_targets = np.asarray(metrics["test_targets"])
    predictions = np.asarray(metrics["predictions"])
    correct_mask = test_targets == predictions
    bins = np.linspace(0, 1, 11)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(confidences[correct_mask], bins=bins, alpha=0.75, label="Correct", color="#06A77D", edgecolor="white")
    axis.hist(confidences[~correct_mask], bins=bins, alpha=0.75, label="Incorrect", color="#E63946", edgecolor="white")
    axis.set_xlabel("Prediction Confidence", fontweight="bold")
    axis.set_ylabel("Samples", fontweight="bold")
    axis.set_title("Confidence Distribution", fontweight="bold")
    axis.set_xlim([0, 1])
    axis.grid(True, alpha=0.3, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_training_tables(
    history: dict[str, list[float]],
    metrics: dict[str, object],
    output_dir: Path,
) -> None:
    pd.DataFrame({
        "epoch": np.arange(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "train_accuracy": history["train_accuracy"],
        "validation_loss": history["validation_loss"],
        "validation_accuracy": history["validation_accuracy"],
    }).to_csv(output_dir / "training_history.csv", index=False)

    class_names = np.asarray(metrics["class_names"])
    pd.DataFrame({
        "class": class_names,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "support": metrics["support"],
    }).to_csv(output_dir / "class_metrics.csv", index=False)

    test_targets = np.asarray(metrics["test_targets"], dtype=int)
    predictions = np.asarray(metrics["predictions"], dtype=int)
    confidences = np.asarray(metrics["confidences"], dtype=float)
    pd.DataFrame({
        "actual_index": test_targets,
        "predicted_index": predictions,
        "actual_class": class_names[test_targets],
        "predicted_class": class_names[predictions],
        "confidence": confidences,
        "correct": test_targets == predictions,
    }).to_csv(output_dir / "predictions.csv", index=False)