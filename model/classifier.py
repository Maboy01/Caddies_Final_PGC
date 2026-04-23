"""
model/classifier.py — Arquitectura CNN+LSTM para clasificación de swings de golf.

El modelo combina dos tipos de redes neuronales:

    1. CNN (MobileNetV3-Small): Extrae características visuales de cada frame
       individualmente. Recibe una imagen y produce un vector de 576 números
       que resume qué hay en esa imagen (forma del cuerpo, posición del palo, etc.)

    2. LSTM Bidireccional: Analiza la secuencia temporal de esos vectores CNN
       para capturar el movimiento a lo largo del swing. Al ser bidireccional,
       lee la secuencia de adelante hacia atrás Y de atrás hacia adelante,
       lo que le permite entender mejor el contexto temporal.

Flujo de datos:
    (batch, T, C, H, W) → CNN por frame → (batch, T, 576) → LSTM → (batch, 256) → Lineal → logits
"""
from __future__ import annotations

import torchvision.models as tv_models
from torch import nn
import torch


class CnnLstmClassifier(nn.Module):
    """
    Clasificador de swings de golf que combina CNN y LSTM.

    Arquitectura:
        - CNN: MobileNetV3-Small preentrenado en ImageNet (sin la cabeza de clasificación).
               Produce 576 características por frame.
        - LSTM: 2 capas bidireccionales, hidden_size=128 → salida de 256 por paso.
        - Dropout: 0.3 dentro del LSTM y 0.4 antes del clasificador final.
        - Clasificador: Capa lineal 256 → num_classes.

    Args:
        num_classes : Número de clases a predecir (ej. 2 para wood/iron)
        hidden_size : Dimensión del estado oculto del LSTM (por dirección)
    """

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()

        # Carga MobileNetV3-Small con pesos de ImageNet y descarta su clasificador original
        backbone = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # Conserva solo el extractor de características (todo excepto la última capa lineal)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out_size = 576  # Dimensión del vector de características de MobileNetV3-Small

        # LSTM bidireccional para modelar la dinámica temporal del swing
        # bidireccional=True duplica el tamaño de salida: hidden_size*2
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=hidden_size,
            num_layers=2,           # 2 capas apiladas para mayor capacidad
            batch_first=True,       # Primer dim del tensor es el batch, no el tiempo
            bidirectional=True,     # Lee la secuencia en ambas direcciones
            dropout=0.3,            # Dropout entre capas del LSTM para regularización
        )

        self.dropout = nn.Dropout(0.4)  # Regularización antes del clasificador final

        # Capa lineal final: proyecta los 256 features del LSTM a las clases
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.

        Args:
            sequences: Tensor de forma (batch_size, sequence_length, C, H, W)
                       donde C=3 (RGB), H y W son las dimensiones del frame.

        Retorna logits sin activar de forma (batch_size, num_classes).
        Para obtener probabilidades, aplica softmax a la salida.

        Pasos internos:
            1. Reorganiza el batch para procesar todos los frames juntos con la CNN.
            2. La CNN extrae un vector de 576 características por frame.
            3. Reorganiza de vuelta a secuencias temporales.
            4. El LSTM procesa cada secuencia y produce un vector por paso de tiempo.
            5. Se toma solo el último paso de tiempo como representación del swing completo.
            6. Dropout + clasificador lineal → logits finales.
        """
        batch_size, sequence_length, channels, height, width = sequences.shape

        # Fusiona el batch y la dimensión temporal para procesar todos los frames a la vez
        # (batch_size * sequence_length, C, H, W)
        frame_batch = sequences.reshape(batch_size * sequence_length, channels, height, width)

        # Extrae características de cada frame con la CNN
        frame_features = self.cnn(frame_batch)
        frame_features = frame_features.flatten(start_dim=1)  # (N*T, 576)

        # Restaura la dimensión temporal: (batch_size, sequence_length, 576)
        temporal_features = frame_features.reshape(batch_size, sequence_length, -1)

        # El LSTM procesa la secuencia completa
        # lstm_output: (batch_size, sequence_length, hidden_size*2)
        lstm_output, _hidden_state = self.lstm(temporal_features)

        # Toma el último paso de tiempo como resumen del swing completo
        final_features = lstm_output[:, -1, :]  # (batch_size, hidden_size*2)

        # Aplica dropout y proyecta a logits de clases
        return self.classifier(self.dropout(final_features))
