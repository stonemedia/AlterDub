from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class VCDecoder(nn.Module):
    """
    Voice Conversion Decoder.

    Inputs:
      - content_repr: (B, T, content_dim)
      - f0:            (B, T)
      - spk_emb:       (B, spk_emb_dim)

    Output:
      - mel_hat:       (B, T, n_mels)
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()

        n_mels = int(model_cfg["n_mels"])
        content_dim = int(model_cfg["content_dim"])
        spk_emb_dim = int(model_cfg["spk_emb_dim"])
        f0_emb_dim = int(model_cfg["f0_emb_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        num_layers = int(model_cfg["num_decoder_layers"])
        dropout = float(model_cfg.get("dropout", 0.1))

        # Causal switch
        causal = bool(model_cfg.get("causal", False))
        bidirectional = not causal

        self.n_mels = n_mels
        self.content_dim = content_dim
        self.spk_emb_dim = spk_emb_dim
        self.f0_emb_dim = f0_emb_dim
        self.hidden_dim = hidden_dim

        # F0 embedding: scalar f0 -> vector of dim f0_emb_dim
        self.f0_emb = nn.Linear(1, f0_emb_dim)

        # Input fusion: content + f0_emb + spk_emb
        fused_dim = content_dim + f0_emb_dim + spk_emb_dim
        self.input_proj = nn.Linear(fused_dim, hidden_dim)

        # RNN core
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(gru_out_dim, n_mels)

    def forward(
        self,
        content_repr: torch.Tensor,
        f0: torch.Tensor,
        spk_emb: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            content_repr: FloatTensor (B, T, content_dim)
            f0:           FloatTensor (B, T)
            spk_emb:      FloatTensor (B, spk_emb_dim)
            lengths:      LongTensor (B,), optional
            mask:         FloatTensor (B, T), optional

        Returns:
            mel_hat:      FloatTensor (B, T, n_mels)
        """
        B, T, _ = content_repr.shape

        # F0 embedding
        f0_input = f0.unsqueeze(-1)                 # (B, T, 1)
        f0_emb = self.f0_emb(f0_input)              # (B, T, f0_emb_dim)

        # Broadcast speaker embedding across time
        spk_emb_expanded = spk_emb.unsqueeze(1).expand(-1, T, -1)

        # Fuse features
        x = torch.cat([content_repr, f0_emb, spk_emb_expanded], dim=-1)
        x = self.input_proj(x)

        # GRU
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths=lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            out_packed, _ = self.gru(x_packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed,
                batch_first=True,
                total_length=T,
            )
        else:
            out, _ = self.gru(x)

        mel_hat = self.output_proj(out)

        if mask is not None:
            mel_hat = mel_hat * mask.unsqueeze(-1)

        return mel_hat
