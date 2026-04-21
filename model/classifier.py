from __future__ import annotations

import torchvision.models as tv_models
from torch import nn
import torch


class CnnLstmClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out_size = 576

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = sequences.shape
        frame_batch = sequences.reshape(batch_size * sequence_length, channels, height, width)
        frame_features = self.cnn(frame_batch)
        frame_features = frame_features.flatten(start_dim=1)
        temporal_features = frame_features.reshape(batch_size, sequence_length, -1)
        lstm_output, _hidden_state = self.lstm(temporal_features)
        final_features = lstm_output[:, -1, :]
        return self.classifier(self.dropout(final_features))