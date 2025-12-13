from typing import Dict, Any

import torch
import torch.nn as nn


class ContentEncoderWrapper(nn.Module):
    """
    v0 Content Encoder Wrapper.

    For now, this is a simple linear projection from mel features to a
    'content representation' space of dimension content_dim.

    Later, this can be swapped to a pretrained HuBERT/ContentVec encoder
    that consumes raw waveform. The rest of the pipeline (decoder, trainer)
    doesn't need to change.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()

        n_mels = int(model_cfg["n_mels"])
        content_dim = int(model_cfg["content_dim"])

        # Simple per-frame projection: (B, T, n_mels) -> (B, T, content_dim)
        self.proj = nn.Linear(n_mels, content_dim)

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            mel: FloatTensor of shape (B, T, n_mels)
            lengths: (B,) optional, not used in v0
            mask: (B, T) optional, not used in v0

        Returns:
            content_repr: FloatTensor of shape (B, T, content_dim)
        """
        content_repr = self.proj(mel)  # (B, T, content_dim)
        return content_repr
