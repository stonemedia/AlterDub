# -*- coding: utf-8 -*-
# Indentation: 4 spaces only (NO TABS)

import os
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module4_vc_training.utils.losses import VCLoss
from module4_vc_training.utils.checkpoints import save_checkpoint
from module4_vc_training.utils.logging_utils import create_tensorboard_writer

from contextlib import nullcontext

try:
    from torch.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


class VCTrainer:
    """
    Training loop for Module 4 (Content Encoder VC, v0).
    AMP-enabled, resume-safe, interrupt-safe.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        run_dir: str,
        logger,
        content_encoder: nn.Module,
        decoder: nn.Module,
        speaker_embedding: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.logger = logger

        self.content_encoder = content_encoder
        self.decoder = decoder
        self.speaker_embedding = speaker_embedding

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = VCLoss(cfg["loss"]).to(device)

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Freeze content encoder
        for p in self.content_encoder.parameters():
            p.requires_grad = False
        self.content_encoder.eval()

        # Optimizer
        params = list(self.decoder.parameters()) + list(self.speaker_embedding.parameters())
        lr = float(cfg["training"]["learning_rate"])
        wd = float(cfg["training"].get("weight_decay", 0.0))
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        # AMP (only active on CUDA)
        self.use_amp = bool(cfg["training"].get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp) if (GradScaler is not None) else None

        # TensorBoard
        self.tb_writer = None
        if cfg["logging"].get("use_tensorboard", True):
            self.tb_writer = create_tensorboard_writer(
                cfg["logging"]["log_dir"],
                os.path.basename(run_dir),
            )

        # Save config snapshot
        with open(os.path.join(run_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def _log_scalars(self, scalars: Dict[str, float], step: int, prefix: str) -> None:
        msg = " ".join([f"{k}={v:.6f}" for k, v in scalars.items()])
        self.logger.info(f"{prefix} step={step} {msg}")

        if self.tb_writer is not None:
            for k, v in scalars.items():
                self.tb_writer.add_scalar(f"{prefix}/{k}", v, step)

    @torch.no_grad()
    def validate(self) -> float:
        self.decoder.eval()
        self.speaker_embedding.eval()
        self.content_encoder.eval()

        total_loss = 0.0
        count = 0

        for batch in self.val_loader:
            mel = batch["mel"].to(self.device)
            f0 = batch["f0"].to(self.device)
            spk_id = batch["spk_id"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            mask = batch["mask"].to(self.device)

            content_repr = self.content_encoder(mel, lengths=lengths, mask=mask)
            spk_emb = self.speaker_embedding(spk_id)
            mel_hat = self.decoder(content_repr, f0, spk_emb, lengths=lengths, mask=mask)

            losses = self.loss_fn(mel_hat, mel, mask=mask)
            total_loss += losses["total_loss"].item()
            count += 1

        return total_loss / max(count, 1)

    def _save(self, name: str) -> None:
        ckpt_path = os.path.join(self.run_dir, f"{name}.pt")
        save_checkpoint(
            path=ckpt_path,
            step=self.step,
            epoch=self.epoch,
            model_state={
                "content_encoder": self.content_encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "speaker_embedding": self.speaker_embedding.state_dict(),
            },
            optimizer_state=self.optimizer.state_dict(),
            best_val_loss=self.best_val_loss,
            cfg=self.cfg,
        )
        self.logger.info(f"[CKPT] Saved: {ckpt_path}")

    def train(self) -> None:
        max_steps = int(self.cfg["training"].get("max_steps", 200000))
        log_every = int(self.cfg["training"].get("log_every_steps", 50))
        val_every = int(self.cfg["training"].get("val_every_steps", 1000))
        save_every = int(self.cfg["training"].get("save_every_steps", 1000))
        grad_clip = float(self.cfg["training"].get("grad_clip", 1.0))

        self.logger.info("Starting training...")
        self.decoder.train()
        self.speaker_embedding.train()

        try:
            while self.step < max_steps:
                self.epoch += 1

                for batch in self.train_loader:
                    self.step += 1
                    if self.step > max_steps:
                        break

                    mel = batch["mel"].to(self.device)
                    f0 = batch["f0"].to(self.device)
                    spk_id = batch["spk_id"].to(self.device)
                    lengths = batch["lengths"].to(self.device)
                    mask = batch["mask"].to(self.device)

                    # Content encoder (frozen)
                    with torch.no_grad():
                        content_repr = self.content_encoder(mel, lengths=lengths, mask=mask)

                    # AMP context
                    amp_ctx = autocast(device_type="cuda", enabled=self.use_amp) if autocast is not None else nullcontext()

                    with amp_ctx:
                        spk_emb = self.speaker_embedding(spk_id)
                        mel_hat = self.decoder(
                            content_repr,
                            f0,
                            spk_emb,
                            lengths=lengths,
                            mask=mask,
                        )
                        losses = self.loss_fn(mel_hat, mel, mask=mask)
                        loss = losses["total_loss"]

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(loss).backward()

                        if grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                list(self.decoder.parameters()) +
                                list(self.speaker_embedding.parameters()),
                                max_norm=grad_clip,
                            )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()

                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(self.decoder.parameters()) +
                                list(self.speaker_embedding.parameters()),
                                max_norm=grad_clip,
                            )

                        self.optimizer.step()

                    # Logging
                    if self.step % log_every == 0:
                        self._log_scalars(
                            {
                                "total_loss": loss.item(),
                                "mel_loss": losses["mel_loss"].item(),
                            },
                            self.step,
                            prefix="train",
                        )

                    # Validation
                    if self.step % val_every == 0:
                        val_loss = self.validate()
                        self._log_scalars(
                            {"val_loss": val_loss},
                            self.step,
                            prefix="val",
                        )

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save("best")

                        # IMPORTANT: switch back to train mode after validation
                        self.decoder.train()
                        self.speaker_embedding.train()
                        self.content_encoder.eval()

                    # Periodic checkpoint
                    if self.step % save_every == 0:
                        self._save("last")

            self.logger.info("Training complete.")

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Saving interrupt checkpoint...")
            self._save("last_interrupt")

        finally:
            if self.tb_writer is not None:
                self.tb_writer.close()
