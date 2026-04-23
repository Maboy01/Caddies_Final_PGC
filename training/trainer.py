"""
training/trainer.py — Lógica completa de entrenamiento y evaluación del modelo.

Contiene:
    - División de datos en train/test con estratificación
    - Cálculo de pesos por clase para manejar desbalance
    - Creación de DataLoaders de PyTorch
    - Loop de entrenamiento por época
    - Evaluación del modelo sin gradientes
    - Función principal que orquesta todo el entrenamiento con early stopping
"""
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

RANDOM_STATE = 42   # Semilla para reproducibilidad en la división de datos
TEST_SIZE = 0.3     # 30% de los datos se reservan para evaluación final


def split_indices(targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide los índices del dataset en conjuntos de entrenamiento y prueba.

    Usa estratificación cuando es posible (al menos 2 muestras por clase),
    para que la proporción de clases sea similar en ambos conjuntos.
    Si una clase tiene muy pocos ejemplos, usa división aleatoria simple.

    Args:
        targets: Tensor con las etiquetas numéricas de cada muestra

    Retorna una tupla (train_indexes, test_indexes) de arrays numpy.
    """
    target_array = targets.numpy()
    class_counts = pd.Series(target_array).value_counts()
    class_count = len(class_counts)

    # Verifica si es posible estratificar (cada clase necesita ≥2 muestras)
    can_stratify = class_count > 1 and class_counts.min() >= 2

    if can_stratify:
        # Asegura que haya al menos un ejemplo de cada clase en el set de prueba
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
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Los pesos se usan en CrossEntropyLoss para penalizar más los errores
    en clases con pocas muestras, compensando el desbalance del dataset.

    Fórmula: weight[c] = total_muestras / (conteo[c] * num_classes)

    Args:
        targets     : Etiquetas del conjunto de entrenamiento
        num_classes : Número total de clases
        device      : Dispositivo donde colocar el tensor resultante

    Retorna un tensor de forma (num_classes,) con los pesos por clase.
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # Evita división por cero si falta alguna clase
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def make_loader(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    indexes: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Crea un DataLoader de PyTorch para un subconjunto de los datos.

    Args:
        sequences  : Tensor con todas las secuencias de frames (N, T, C, H, W)
        targets    : Tensor con todas las etiquetas (N,)
        indexes    : Índices del subconjunto a usar (train o test)
        batch_size : Tamaño del mini-batch
        shuffle    : True para entrenamiento, False para evaluación

    Retorna un DataLoader que itera sobre el subconjunto indicado.
    """
    dataset = TensorDataset(sequences[indexes], targets[indexes])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Ejecuta una época completa de entrenamiento (forward + backprop + update).

    Por cada mini-batch:
        1. Mueve los datos al dispositivo (GPU/CPU)
        2. Pone los gradientes a cero
        3. Propaga hacia adelante para obtener logits
        4. Calcula la pérdida (CrossEntropy)
        5. Propaga el gradiente hacia atrás
        6. Actualiza los pesos del modelo

    Args:
        model     : Modelo a entrenar
        loader    : DataLoader del conjunto de entrenamiento
        criterion : Función de pérdida (CrossEntropyLoss con pesos)
        optimizer : Optimizador (Adam)
        device    : Dispositivo de cómputo

    Retorna (pérdida_promedio, precisión_promedio) sobre toda la época.
    """
    model.train()  # Activa modo entrenamiento (dropout activo, BN en modo train)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_sequences, batch_targets in loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad(set_to_none=True)        # Limpia gradientes anteriores
        logits = model(batch_sequences)              # Forward pass
        loss = criterion(logits, batch_targets)      # Calcula pérdida
        loss.backward()                              # Backpropagation
        optimizer.step()                             # Actualiza pesos

        batch_size = batch_targets.size(0)
        total_loss += float(loss.item()) * batch_size  # Acumula pérdida ponderada por tamaño
        predictions = torch.argmax(logits, dim=1)      # Clase predicha = índice del logit máximo
        correct_predictions += int((predictions == batch_targets).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, correct_predictions / total_samples


@torch.no_grad()  # Desactiva el cálculo de gradientes (ahorra memoria y acelera)
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo sobre un conjunto de datos sin actualizar pesos.

    Además de pérdida y precisión, recopila las predicciones individuales
    y la confianza (probabilidad máxima) para generar métricas detalladas.

    Args:
        model     : Modelo entrenado
        loader    : DataLoader del conjunto a evaluar
        criterion : Función de pérdida
        device    : Dispositivo de cómputo

    Retorna una tupla con:
        - pérdida promedio
        - precisión promedio
        - predicciones (array de índices de clase)
        - etiquetas reales (array de índices de clase)
        - confianzas (probabilidad de la clase predicha)
    """
    model.eval()  # Activa modo evaluación (dropout desactivado)
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

        # Convierte logits a probabilidades con softmax
        probabilities = torch.softmax(logits, dim=1)
        # La confianza es la probabilidad de la clase con mayor score
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
    """
    Orquesta el entrenamiento completo del modelo CNN+LSTM.

    Características:
        - División train/test estratificada (70/30)
        - CrossEntropyLoss con pesos por clase para manejar desbalance
        - Optimizador Adam con learning rate adaptativo (ReduceLROnPlateau)
        - Early stopping: detiene el entrenamiento si la pérdida de validación
          no mejora por 6 épocas consecutivas
        - Guarda automáticamente el mejor modelo (menor validation loss)

    Args:
        sequences   : Tensor con todas las secuencias de frames
        targets     : Tensor con todas las etiquetas
        num_classes : Número de clases del problema
        args        : Hiperparámetros (epochs, batch_size, learning_rate)
        device      : Dispositivo de cómputo (CPU/GPU)
        output_dir  : Carpeta donde guardar el mejor modelo temporal

    Retorna:
        model        : Modelo cargado con los mejores pesos encontrados
        history      : Diccionario con listas de métricas por época
        predictions  : Predicciones finales sobre el set de prueba
        test_targets : Etiquetas reales del set de prueba
        confidences  : Confianzas de las predicciones finales
    """
    # --- Preparación de datos ---
    train_indexes, test_indexes = split_indices(targets)
    train_loader = make_loader(sequences, targets, train_indexes, args.batch_size, shuffle=True)
    test_loader  = make_loader(sequences, targets, test_indexes,  args.batch_size, shuffle=False)

    # --- Inicialización del modelo y optimizadores ---
    model = CnnLstmClassifier(num_classes=num_classes).to(device)
    class_weights = build_class_weights(targets[train_indexes], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Reduce el learning rate a la mitad si la val_loss no mejora en 3 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    # --- Estado de early stopping ---
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = 6  # Número de épocas sin mejora antes de parar

    # Historial de métricas para generar gráficos posteriormente
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    # --- Loop de entrenamiento ---
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        validation_loss, validation_accuracy, _p, _t, _c = evaluate_model(
            model, test_loader, criterion, device,
        )

        # Guarda métricas del epoch actual
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        # Ajusta el learning rate según el progreso de la validación
        scheduler.step(validation_loss)

        if validation_loss < best_val_loss:
            # Nuevo mejor modelo: guarda sus pesos
            best_val_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"   Early stopping en época {epoch}")
                break  # Detiene el entrenamiento por falta de mejora

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )

    # --- Evaluación final con el mejor modelo ---
    # Carga el checkpoint con la mejor val_loss registrada durante el entrenamiento
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    _loss, _accuracy, predictions, test_targets, confidences = evaluate_model(
        model, test_loader, criterion, device,
    )
    return model, history, predictions, test_targets, confidences
