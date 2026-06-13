"""PatchTST policy encoder with multi-head action outputs."""

from __future__ import annotations

from ai.models.patchtst_model import torch, nn


ACTION_CLASSES = {
    0: "NO_ACTION",
    1: "BUY",
    2: "HOLD_POSITION",
    3: "TAKE_PROFIT_PARTIAL",
    4: "TAKE_PROFIT_FULL",
    5: "CUT_LOSS",
    6: "TRAILING_EXIT",
}
ACTION_TO_CLASS = {name: idx for idx, name in ACTION_CLASSES.items()}


if nn is not None:

    class PatchTSTPolicyModel(nn.Module):
        def __init__(
            self,
            num_features: int,
            sequence_length: int = 96,
            patch_length: int = 8,
            d_model: int = 64,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            num_actions: int = 7,
        ) -> None:
            super().__init__()
            self.num_features = num_features
            self.sequence_length = sequence_length
            self.patch_length = patch_length
            self.num_patches = sequence_length // patch_length
            self.d_model = d_model
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
            self.action_head = nn.Linear(d_model, num_actions)
            self.return_head = nn.Linear(d_model, 1)
            self.risk_head = nn.Linear(d_model, 1)
            self.holding_time_head = nn.Linear(d_model, 1)
            self.reconstruction_head = nn.Linear(d_model, num_features * patch_length)

        def encode_patches(self, x):
            batch, _, num_features = x.shape
            x = x[:, : self.num_patches * self.patch_length, :]
            patches = x.reshape(batch, self.num_patches, self.patch_length * num_features)
            encoded = self.encoder(self.patch_proj(patches) + self.pos_embedding)
            return encoded

        def encode(self, x):
            encoded = self.encode_patches(x)
            return self.norm(encoded.mean(dim=1))

        def forward(self, x):
            pooled = self.encode(x)
            return {
                "action_logits": self.action_head(pooled),
                "expected_return": self.return_head(pooled).squeeze(-1),
                "risk_score": torch.sigmoid(self.risk_head(pooled).squeeze(-1)),
                "holding_time_score": torch.sigmoid(self.holding_time_head(pooled).squeeze(-1)),
            }

        def reconstruct_patches(self, x):
            encoded = self.encode_patches(x)
            return self.reconstruction_head(encoded)

else:
    PatchTSTPolicyModel = None
