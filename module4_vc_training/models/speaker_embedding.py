from typing import Dict, Any

import torch
import torch.nn as nn


class SpeakerEmbedding(nn.Module):
    """
    Simple speaker embedding lookup.

    Maps integer speaker IDs to dense vectors of dimension spk_emb_dim.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()

        num_speakers = int(model_cfg["num_speakers"])
        spk_emb_dim = int(model_cfg["spk_emb_dim"])

        self.embedding = nn.Embedding(num_embeddings=num_speakers, embedding_dim=spk_emb_dim)

    def forward(self, spk_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spk_ids: LongTensor of shape (B,)

        Returns:
            spk_emb: FloatTensor of shape (B, spk_emb_dim)
        """
        return self.embedding(spk_ids)
