import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)


class DiscriminatorP(nn.Module):
    """Period discriminator (2D conv over reshaped waveform)."""
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1,  32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128,256, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(256,512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512,1024,(kernel_size, 1), (1, 1), padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), (1, 1), padding=(1, 0)))

        self.convs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # x: (B, 1, T)
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode='reflect')
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)  # (B,1,T//p,p)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def remove_weight_norm(self):
        for l in self.convs:
            try: remove_weight_norm(l)
            except: pass
        try: remove_weight_norm(self.conv_post)
        except: pass


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            r, fr = d(y)
            g, fg = d(y_hat)
            y_d_rs.append(r)
            y_d_gs.append(g)
            fmap_rs.append(fr)
            fmap_gs.append(fg)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """Scale discriminator (1D conv stack)."""
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1,  128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128,128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128,256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256,512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512,1024,41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024,1024,41,1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024,1024,5,1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

        self.convs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def remove_weight_norm(self):
        for l in self.convs:
            try: remove_weight_norm(l)
            except: pass
        try: remove_weight_norm(self.conv_post)
        except: pass


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        yr = y
        yg = y_hat
        for i, d in enumerate(self.discriminators):
            r, fr = d(yr)
            g, fg = d(yg)
            y_d_rs.append(r)
            y_d_gs.append(g)
            fmap_rs.append(fr)
            fmap_gs.append(fg)
            if i < len(self.meanpools):
                yr = self.meanpools[i](yr)
                yg = self.meanpools[i](yg)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

