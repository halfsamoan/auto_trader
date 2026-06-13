"""Small local PatchTST-lite model for sequence classification/regression."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except Exception:
        return None, None
    return torch, nn


torch, nn = require_torch()


if nn is not None:

    class PatchTSTLite(nn.Module):
        def __init__(
            self,
            num_features: int,
            sequence_length: int = 96,
            patch_length: int = 8,
            d_model: int = 64,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            num_classes: int = 1,
        ) -> None:
            super().__init__()
            self.num_features = num_features
            self.sequence_length = sequence_length
            self.patch_length = patch_length
            self.num_classes = num_classes
            self.num_patches = sequence_length // patch_length
            self.patch_proj = nn.Linear(num_features * patch_length, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.class_head = nn.Linear(d_model, num_classes)
            self.return_head = nn.Linear(d_model, 1)
            self.risk_head = nn.Linear(d_model, 1)

        def forward(self, x):
            batch, seq_len, num_features = x.shape
            x = x[:, : self.num_patches * self.patch_length, :]
            x = x.reshape(batch, self.num_patches, self.patch_length * num_features)
            encoded = self.encoder(self.patch_proj(x) + self.pos_embedding)
            pooled = self.norm(encoded.mean(dim=1))
            class_output = self.class_head(pooled)
            if self.num_classes == 1:
                class_output = class_output.squeeze(-1)
            return {
                "logit": class_output,
                "expected_return": self.return_head(pooled).squeeze(-1),
                "risk_score": torch.sigmoid(self.risk_head(pooled).squeeze(-1)),
            }

else:
    PatchTSTLite = None
