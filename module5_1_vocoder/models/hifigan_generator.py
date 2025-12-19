import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                stride=1,
                dilation=d,
                padding=get_padding(kernel_size, d)
            )) for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )) for _ in dilations
        ])

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Module):
    """HiFi-GAN Generator (V1 style), adapted for hop_length=80.

    Input:  mel (B, 80, T_mel)
    Output: audio (B, 1, T_audio) where T_audio = T_mel * hop_length
    """
    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates=(5, 4, 4),
        upsample_kernel_sizes=(10, 8, 8),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        assert len(upsample_rates) == len(upsample_kernel_sizes)
        self.n_mels = n_mels
        self.hop_length = int(math.prod(list(upsample_rates)))

        self.conv_pre = weight_norm(nn.Conv1d(n_mels, upsample_initial_channel, kernel_size=7, stride=1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        cur_ch = upsample_initial_channel
        for (u, k) in zip(upsample_rates, upsample_kernel_sizes):
            # Length-exact ConvTranspose config:
            # p = ceil((k-u)/2), op = u + 2p - k  ensures out_len = in_len * u
            p = (k - u + 1) // 2
            op = u + 2 * p - k
            if op < 0 or op >= u:
                raise ValueError(f'Invalid output_padding computed: u={u}, k={k}, p={p}, op={op}')

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        cur_ch,
                        cur_ch // 2,
                        kernel_size=k,
                        stride=u,
                        padding=p,
                        output_padding=op,
                    )
                )
            )
            cur_ch = cur_ch // 2

            for (rk, rd) in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock1(cur_ch, kernel_size=rk, dilations=rd))

        self.conv_post = weight_norm(nn.Conv1d(cur_ch, 1, kernel_size=7, stride=1, padding=3))

        self.ups.apply(init_weights)
        self.conv_pre.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.num_upsamples = len(self.ups)
        self.num_kernels = len(resblock_kernel_sizes)

    def forward(self, mel):
        x = self.conv_pre(mel)
        r = 0
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for _ in range(self.num_kernels):
                rb = self.resblocks[r]
                r += 1
                out = rb(x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for l in self.ups:
            remove_weight_norm(l)
        for rb in self.resblocks:
            rb.remove_weight_norm()

