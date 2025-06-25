import math

import paddle


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(tuple(timesteps.shape)) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = paddle.exp(x=paddle.arange(dtype="float32", end=half_dim) * -emb)
    emb = emb.to(device=timesteps.place)
    emb = timesteps.astype(dtype="float32")[:, None] * emb[None, :]
    emb = paddle.concat(x=[paddle.sin(x=emb), paddle.cos(x=emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = paddle.nn.functional.pad(
            x=emb, pad=(0, 1, 0, 0)
        )
    return emb


def nonlinearity(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def Normalize(in_channels):
    return paddle.nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        epsilon=1e-06,
        weight_attr=True,
        bias_attr=True,
    )


class Upsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, output_size=None):
        if output_size is not None:
            x = paddle.nn.functional.interpolate(
                x=x, size=output_size, mode="nearest"
            )
        else:
            x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = paddle.nn.functional.pad(
                x=x, pad=pad, mode="constant", value=0
            )
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(
                x=x, kernel_size=2, stride=2, exclusive=False
            )
        return x


class ResnetBlock(paddle.nn.Layer):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.temb_proj = paddle.nn.Linear(
            in_features=temb_channels, out_features=out_channels
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = tuple(q.shape)
        q = q.reshape(b, c, h * w)
        q = q.transpose(perm=[0, 2, 1])
        k = k.reshape(b, c, h * w)
        w_ = paddle.bmm(x=q, y=k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.transpose(perm=[0, 2, 1])
        h_ = paddle.bmm(x=v, y=w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Model(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = (
            config.model.ch,
            config.model.out_ch,
            tuple(config.model.ch_mult),
        )
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        if config.model.type == "bayesian":
            self.logvar = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=num_timesteps)
            )
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temb = paddle.nn.Layer()
        self.temb.dense = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(in_features=self.ch, out_features=self.temb_ch),
                paddle.nn.Linear(in_features=self.temb_ch, out_features=self.temb_ch),
            ]
        )
        self.conv_in = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = paddle.nn.LayerList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.append(up)
        self.up = self.up[::-1]
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    paddle.concat(x=[h, hs.pop()], axis=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                if self.config.data.dataset == 'sst':
                    h = self.up[i_level].upsample(h, self.config.model.output_size[i_level-1])
                else:
                    h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
