import os
import math
import torch
import torch.nn as nn
import numpy as np

def sqrt(x):
    return np.sqrt(x, dtype=np.float32)

# ===== Forced weight normalization =====

def normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


class Conv2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        w = torch.randn(C_out, C_in, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight) / sqrt(fan_in)
        x = nn.functional.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding)
        return x


class Linear(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        w = torch.randn(C_out, C_in)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight) / sqrt(fan_in)
        x = nn.functional.linear(x, w, bias=None)
        return x


def GetConv2d(C_in, C_out, kernel_size, stride=1, padding=0):
    return Conv2d(C_in, C_out, kernel_size, stride, padding)


def GetLinear(C_in, C_out):
    return Linear(C_in, C_out)

# ===== Magnitude-preserving fixed-function layers =====

class MP_Fourier(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(embedding_size), requires_grad=False)
        self.phases      = nn.Parameter(torch.rand(embedding_size), requires_grad=False)

    def forward(self, a):
        b = (2 * np.pi) * (a[:, None] * self.frequencies[None, :] + self.phases[None, :])
        b = torch.cos(b) * sqrt(2.)
        return b


class MP_SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x) / 0.596


class MP_Sum(nn.Module):
    def __init__(self, b_contribution=0.3):
        super().__init__()
        self.b_contribution = b_contribution
        self.a_contribution = 1.0 - b_contribution
        self.div = sqrt(self.a_contribution ** 2 + self.b_contribution ** 2)

    def forward(self, a, b):
        return (self.a_contribution * a + self.b_contribution * b) / self.div


class MP_Cat(nn.Module):
    def __init__(self, b_contribution=0.5):
        super().__init__()
        self.b_contribution = b_contribution
        self.a_contribution = 1.0 - b_contribution
        self.div = sqrt(self.a_contribution ** 2 + self.b_contribution ** 2)

    def forward(self, a, b):
        Na = a.shape[1]
        Nb = b.shape[1]
        c = torch.cat([self.a_contribution * a / sqrt(Na), self.b_contribution * b / sqrt(Nb)], dim=1)
        return c * sqrt(Na + Nb) / self.div


class Gain(nn.Module):
    def __init__(self, init='zero'):
        super().__init__()
        if init == 'one':
            self.g = nn.Parameter(torch.tensor(1.0))
        else:
            assert init == 'zero'
            self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.g

# ===== Embedding =====

class Embedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.fourier = MP_Fourier(embedding_size=n_channels // 4)
        self.linear1 = GetLinear(n_channels // 4, n_channels)
        self.act = MP_SiLU()

    def forward(self, c_noise):
        emb = self.fourier(c_noise)
        emb = self.act(self.linear1(emb))
        return emb


class Reweighting(nn.Module):
    def __init__(self, n_channels=256):
        super().__init__()
        self.fourier = MP_Fourier(embedding_size=n_channels)
        self.linear = GetLinear(n_channels, 1)

    def forward(self, c_noise):
        emb = self.fourier(c_noise)
        emb = self.linear(emb)
        return emb

# ===== Residual blocks =====

def PixNorm(x, dim=1, eps=1e-8):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps)


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(x)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class EncoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout=0.1, down=False):
        super().__init__()
        self.linear = GetLinear(emb_channels, out_channels)
        self.gain = Gain(init='one')

        self.down = Downsample() if down else nn.Identity()
        self.shortcut = GetConv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv1 = GetConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            GetConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.act = MP_SiLU()
        self.outadd = MP_Sum()

    def forward(self, x, emb):
        main = PixNorm(self.shortcut(self.down(x)))
        residual = self.conv1(self.act(main))

        emb = self.gain(self.linear(emb))
        residual = residual * (1 + emb)[:, :, None, None]

        residual = self.conv2(self.act(residual))

        return self.outadd(main, residual)


class DecoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout=0.1, up=False):
        super().__init__()
        self.linear = GetLinear(emb_channels, out_channels)
        self.gain = Gain(init='one')

        self.up = Upsample() if up else nn.Identity()
        self.shortcut = GetConv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv1 = GetConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            GetConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.act = MP_SiLU()
        self.outadd = MP_Sum()

    def forward(self, x, emb):
        main = self.up(x)
        residual = self.conv1(self.act(main))

        emb = self.gain(self.linear(emb))
        residual = residual * (1 + emb)[:, :, None, None]

        residual = self.conv2(self.act(residual))

        main = self.shortcut(main)
        return self.outadd(main, residual)

# ===== Attention block =====

class AttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k):
        super().__init__()
        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k
        assert n_heads * d_k == n_channels

        self.projection = GetConv2d(n_channels, n_channels * 3, kernel_size=1)
        self.output = GetConv2d(n_channels, n_channels, kernel_size=1)

        self.outadd = MP_Sum()
        self.scale = 1 / math.sqrt(math.sqrt(d_k))
        self.n_heads = n_heads
        self.d_k = d_k
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            print(f"{self.n_heads} heads, {self.d_k} channels per head")

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        # (b, c, h, w) -> (b, 3c, h, w)
        h = self.projection(x)

        # (b, 3c, h, w) -> (b, 3c, l) -> (b, l, 3c)
        h = h.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1).contiguous()

        # (b, l, 3c) -> (b, l, n_heads, d_k * 3) -> 3 * (b, l, n_heads, d_k)
        qkv = h.view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = PixNorm(q, dim=3)
        k = PixNorm(k, dim=3)
        v = PixNorm(v, dim=3)
        attn = torch.einsum('bihd,bjhd->bijh', q * self.scale, k * self.scale) # More stable with f16 than dividing afterwards
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # (b, l, n_heads, d_k) -> (b, l, n_heads * d_k) -> (b, n_heads * d_k, l) -> (b, n_heads * d_k, h, w) -> (b, c, h, w)
        res = res.reshape(batch_size, -1, n_channels).permute(0, 2, 1).contiguous()
        res = res.view(batch_size, n_channels, height, width)
        res = self.output(res)
        return self.outadd(x, res)
