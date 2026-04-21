from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.classifier import CnnLstmClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.3


def split_indices(targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    target_array = targets.numpy()
    class_counts = pd.Series(target_array).value_counts()
    class_count = len(class_counts)
    can_stratify = class_count > 1 and class_counts.min() >= 2

    if can_stratify:
        test_count = max(math.ceil(len(target_array) * TEST_SIZE), class_count)
        test_count = min(test_count, len(target_array) - class_count)
        stratify = target_array
    else:
        test_count = max(1, math.ceil(len(target_array) * TEST_SIZE))
        stratify = None
        print("   Using non-stratified split because a class has too few videos.")

    all_indexes = np.arange(len(target_array))
    train_indexes, test_indexes = train_test_split(
        all_indexes,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return train_indexes, test_indexes


def build_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def make_loader(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    indexes: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(sequences[indexes], targets[indexes])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_sequences, batch_targets in loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_sequences)
        loss = criterion(logits, batch_targets)
        loss.backward()
        optimizer.step()

        batch_size = batch_targets.size(0)
        total_loss += float(loss.item()) * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += int((predictions == batch_targets).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, correct_predictions / total_samples


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_confidences = []

    for batch_sequences, batch_targets in loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets = batch_targets.to(device)
        logits = model(batch_sequences)
        loss = criterion(logits, batch_targets)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

        batch_size = batch_targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct_predictions += int((predictions == batch_targets).sum().item())
        total_samples += batch_size
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_targets.extend(batch_targets.cpu().numpy().tolist())
        all_confidences.extend(confidences.cpu().numpy().tolist())

    return (
        total_loss / total_samples,
        correct_predictions / total_samples,
        np.array(all_predictions),
        np.array(all_targets),
        np.array(all_confidences),
    )


def train_cnn_lstm(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[nn.Module, dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray]:
    train_indexes, test_indexes = split_indices(targets)
    train_loader = make_loader(sequences, targets, train_indexes, args.batch_size, shuffle=True)
    test_loader = make_loader(sequences, targets, test_indexes, args.batch_size, shuffle=False)

    model = CnnLstmClassifier(num_classes=num_classes).to(device)
    class_weights = build_class_weights(targets[train_indexes], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = 6
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        validation_loss, validation_accuracy, _p, _t, _c = evaluate_model(
            model, test_loader, criterion, device,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        scheduler.step(validation_loss)
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"   Early stopping en época {epoch}")
                break

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )

    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    _loss, _accuracy, predictions, test_targets, confidences = evaluate_model(
        model, test_loader, criterion, device,
    )
    return model, history, predictions, test_targets, confidences