import torch
from torch import nn
from .blockG import EncoderResBlock, DecoderResBlock, AttentionBlock, Embedding, Reweighting, GetConv2d, Gain, MP_Cat


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout, down, has_attn, attn_channels_per_head):
        super().__init__()
        self.res = EncoderResBlock(in_channels, out_channels, emb_channels, dropout, down)
        if has_attn:
            self.attn = AttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, emb):
        x = self.res(x, emb)
        x = self.attn(x)
        x = torch.clip(x, -256, 256)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout, up, has_attn, attn_channels_per_head):
        super().__init__()
        self.res = DecoderResBlock(in_channels, out_channels, emb_channels, dropout, up)
        if has_attn:
            self.attn = AttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()
        self.up = up

    def forward(self, x, emb):
        x = self.res(x, emb)
        x = self.attn(x)
        x = torch.clip(x, -256, 256)
        return x


class UNetG(nn.Module):
    def __init__(self, image_shape = [3, 32, 32], n_channels = 128,
                 ch_mults = (1, 2, 2, 2),
                 is_attn = (False, True, False, False),
                 attn_channels_per_head = None,
                 dropout = 0.1,
                 n_blocks = 3):
        """
        * `image_shape` is the (channel, height, width) size of images.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `n_channels * ch_mults[i]`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `dropout` is the dropout rate
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        n_resolutions = len(ch_mults)

        self.image_proj = GetConv2d(image_shape[0] + 1, n_channels, kernel_size=3, padding=1)

        # Embedding layers
        emb_channels = n_channels * 4
        self.embedding = Embedding(emb_channels)

        # Down stages
        down = []
        in_channels = n_channels
        h_channels = [n_channels]
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks` at the same resolution
            down.append(EncoderBlock(in_channels, out_channels, emb_channels, dropout, False, is_attn[i], attn_channels_per_head))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 2):
                down.append(EncoderBlock(out_channels, out_channels, emb_channels, dropout, False, is_attn[i], attn_channels_per_head))
                h_channels.append(out_channels)
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(EncoderBlock(out_channels, out_channels, emb_channels, dropout, True, False, 0))
                h_channels.append(out_channels)
            in_channels = out_channels
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle1 = EncoderBlock(out_channels, out_channels, emb_channels, dropout, False, True, attn_channels_per_head)
        self.middle2 = EncoderBlock(out_channels, out_channels, emb_channels, dropout, False, False, 0)

        # Up stages
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks + 1` at the same resolution
            for _ in range(n_blocks):
                up.append(DecoderBlock(in_channels + h_channels.pop(), out_channels, emb_channels, dropout, False, is_attn[i], attn_channels_per_head))
                in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(DecoderBlock(out_channels, out_channels, emb_channels, dropout, True, False, 0))
        assert not h_channels
        self.up = nn.ModuleList(up)
        self.skipcat = MP_Cat()

        # Final convolution layer
        self.final = nn.Sequential(
            GetConv2d(out_channels, image_shape[0], kernel_size=3, padding=1),
            Gain()
        )

    def forward(self, x, t, ret_activation=False):
        if not ret_activation:
            return self.forward_core(x, t)

        activation = {}
        def namedHook(name):
            def hook(module, input, output):
                activation[name] = output
            return hook
        hooks = {}
        no = 0
        for blk in self.up:
            if isinstance(blk, DecoderBlock):
                no += 1
                name = f'out_{no}'
                hooks[name] = blk.register_forward_hook(namedHook(name))

        result = self.forward_core(x, t)
        for name in hooks:
            hooks[name].remove()
        return result, activation

    def forward_core(self, x, t):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        ones_tensor = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)
        x = torch.cat([x, ones_tensor], dim=1)
        x = self.image_proj(x)
        emb = self.embedding(t)

        # `h` will store outputs at each resolution for skip connection
        h = [x]

        for m in self.down:
            x = m(x, emb)
            h.append(x)

        x = self.middle1(x, emb)
        x = self.middle2(x, emb)

        for m in self.up:
            if m.up:
                x = m(x, emb)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = self.skipcat(x, s)
                x = m(x, emb)

        return self.final(x)

    def get_reweighting(self):
        return Reweighting()

    def forward_reweighting(self, MLP, sigma):
        return MLP(sigma.flatten().log() / 4).unsqueeze(-1).unsqueeze(-1)

'''
from model.unetG import UNetG
import torch
net = UNetG()
x = torch.zeros(1, 3, 32, 32)
t = torch.zeros(1,)

net(x, t).shape
sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

>>> 39.542685 M parameters for CIFAR-10 model


net = UNetG(image_shape=[3,64,64], n_channels=192, ch_mults=[1,2,3,4], is_attn=[False,False,True,True], n_blocks=4)
x = torch.zeros(1, 3, 64, 64)
t = torch.zeros(1,)

net(x, t).shape
sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

>>> 279.441253 M parameters for ImageNet-64 model
(becomes exactly 280.2M in Figure 21, after adding 768*1000 class embedding)
'''
