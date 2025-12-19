import torch
import torch.nn.functional as F


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    """LSGAN discriminator loss."""
    loss = 0.0
    losses = []
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        f_loss = torch.mean((df) ** 2)
        loss_i = r_loss + f_loss
        losses.append(loss_i)
        loss = loss + loss_i
    return loss, losses


def generator_loss(disc_fake_outputs):
    """LSGAN generator adversarial loss."""
    loss = 0.0
    losses = []
    for dg in disc_fake_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        losses.append(l)
        loss = loss + l
    return loss, losses


def feature_loss(fmap_real, fmap_fake):
    """Feature matching loss (L1)."""
    loss = 0.0
    for dr, df in zip(fmap_real, fmap_fake):
        for rl, fl in zip(dr, df):
            loss = loss + torch.mean(torch.abs(rl - fl))
    return loss * 2.0

