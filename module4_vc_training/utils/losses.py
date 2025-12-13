from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class VCLoss(nn.Module):
    """
    Module 4 loss (v0):
      - Mel reconstruction L1 loss with padding mask support.
    Later we can add:
      - speaker classification loss
      - adversarial / feature matching losses
      - f0 consistency losses
    """

    def __init__(self, loss_cfg: Dict[str, Any]):
        super().__init__()
        self.w_mel = float(loss_cfg.get("w_mel", 1.0))
        self.l1 = nn.L1Loss(reduction="none")

    def forward(
        self,
        mel_hat: torch.Tensor,
        mel_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel_hat: (B, T, n_mels)
            mel_target: (B, T, n_mels)
            mask: (B, T) float mask, 1.0 for valid frames, 0.0 for padded frames

        Returns:
            dict:
              total_loss, mel_loss
        """
        mel_l1 = self.l1(mel_hat, mel_target)  # (B, T, n_mels)

        if mask is not None:
            mel_l1 = mel_l1 * mask.unsqueeze(-1)
            denom = mask.sum() * mel_hat.shape[-1]
            denom = denom.clamp(min=1.0)
        else:
            denom = torch.tensor(mel_hat.numel(), device=mel_hat.device, dtype=torch.float32)

        mel_loss = mel_l1.sum() / denom
        total_loss = self.w_mel * mel_loss

        return {
            "total_loss": total_loss,
            "mel_loss": mel_loss,
        }
