# modified from https://github.com/lucidrains/bottleneck-transformer-pytorch
# which is translated from tensorflow code
# https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

from torch import nn


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=128
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).view(b, 3 * self.heads, self.dim_head, h * w).chunk(3, dim=1)

        q *= self.scale
        attn = q.transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)

        out = (v @ attn).reshape(b, -1, h, w)
        return out


class PEG(nn.Module):
    def __init__(self, dim, stride):
        super(PEG, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class ResAtt(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            attn_dim_in,
            stride=1,
            heads=4,
            dim_head=128,
            rel_pos_emb=False,
            act_layer=nn.ReLU,
            avg_down=False,
    ):
        super().__init__()
        activation = act_layer(inplace=True)
        norm_layer = nn.BatchNorm2d
        self.inc = dim
        # shortcut

        if avg_down and (stride == 2 or dim != dim_out):
            avg_stride = stride
            if stride == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.shortcut = nn.Sequential(*[
                pool,
                nn.Conv2d(dim, dim_out, 1, stride=1, padding=0, bias=False),
                norm_layer(dim_out)
            ])
        else:
            if stride == 2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(dim, dim_out, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim_out),
                    activation
                )
            elif dim != dim_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(dim, dim_out, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(dim_out),
                    activation
                )
            else:
                self.shortcut = nn.Identity()

        # contraction and expansion
        attn_dim_in = attn_dim_in
        attn_dim_out = heads * dim_head

        self.proj = nn.Sequential(nn.Conv2d(dim, attn_dim_in, 1, bias=False),
                                  PEG(attn_dim_in, stride=stride),
                                  nn.BatchNorm2d(attn_dim_in)
                                  )

        self.net = nn.Sequential(
            activation,
            Attention(
                dim=attn_dim_in,
                heads=heads,
                dim_head=dim_head,
            ),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = activation

    def zero_init_last_bn(self):
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.proj(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)
