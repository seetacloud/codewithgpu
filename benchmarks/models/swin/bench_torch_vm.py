# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Bench SwinTransformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import time

import numpy as np

from dragon.vm import torch
from dragon.vm.torch import nn


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='run training or inference')
    parser.add_argument('--precision', default='float16', help='compute precision')
    parser.add_argument('--device', default=0, type=int, help='compute device')
    parser.add_argument('--model', default='swin_tiny_patch4_window7_224', help='compute model')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
    return parser.parse_args()


def space_to_depth(input, block_size):
    """Rearrange blocks of spatial data into depth."""
    h, w, c = input.size()[1:]
    h1, w1 = h // block_size, w // block_size
    c1 = (block_size ** 2) * c
    input.reshape_((-1, h1, block_size, w1, block_size, c))
    out = input.permute(0, 1, 3, 2, 4, 5)
    input.reshape_((-1, h, w, c))
    return out.reshape_((-1, h1, w1, c1))


def depth_to_space(input, block_size):
    """Rearrange blocks of depth data into spatial."""
    h1, w1, c1 = input.size()[1:]
    h, w = h1 * block_size, w1 * block_size
    c = c1 // (block_size ** 2)
    input.reshape_((-1, h1, w1, block_size, block_size, c))
    out = input.permute(0, 1, 3, 2, 4, 5)
    input.reshape_((-1, h1, w1, c1))
    return out.reshape_((-1, h, w, c))


class RelPosEmbed(nn.Module):
    """Relative position embedding layer."""

    def __init__(self, num_heads, window_size):
        super(RelPosEmbed, self).__init__()
        num_pos = (2 * window_size - 1) ** 2 + 3
        grid = np.arange(window_size)
        pos = np.stack(np.meshgrid(grid, grid, indexing='ij'))
        pos = pos.reshape((2, -1))
        pos = pos[:, :, None] - pos[:, None, :]
        pos += window_size - 1
        pos[0] *= 2 * window_size - 1
        index = pos.sum(0).astype('int64')
        self.register_buffer('index', torch.from_numpy(index))
        self.weight = nn.Parameter(torch.zeros(num_heads, num_pos))
        nn.init.normal_(self.weight, std=.02)

    def forward(self, x):
        if x.device.type == 'mlu' and self.index.dtype == torch.int64:
            self.index.int_()
        return x.add_(self.weight[:, self.index])


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, dim=768, patch_size=16):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        if x.device.type != 'mlu':
            x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """Merge patches to downsample the input."""

    def __init__(self, dim_in, dim_out):
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(4 * dim_in)
        self.reduction = nn.Linear(4 * dim_in, dim_out, bias=False)

    def forward(self, x):
        x = space_to_depth(x, 2)
        return self.reduction(self.norm(x))


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.relative_position = RelPosEmbed(num_heads, window_size)

    def forward(self, x, mask=None):
        num_patches = x.size(1)
        qkv_shape = (-1, num_patches, 3, self.num_heads, self.head_dim)
        qkv = self.qkv(x).reshape_(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0, copy=False)
        attn = q @ k.transpose(-2, -1).mul_(self.scale)
        attn = self.relative_position(attn)
        if mask is not None:
            attn.reshape_(-1, mask.size(1), self.num_heads,
                          num_patches, num_patches).add_(mask)
            attn.reshape_(-1, self.num_heads, num_patches, num_patches)
        attn = nn.functional.softmax(attn, dim=-1, inplace=True)
        return self.proj((attn @ v).transpose(1, 2).flatten_(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4,
        qkv_bias=False,
        drop_path=0,
        downsample=None,
    ):
        super(Block, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, window_size, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path = nn.DropPath(drop_path, inplace=True)
        self.downsample = downsample

    def get_mask(self, resolution):
        index, (height, width) = 0, resolution
        img_mask = np.zeros([1, height, width, 1], 'float32')
        for h, w in itertools.product(
            *[(slice(0, resolution[i] - self.window_size),
               slice(resolution[i] - self.window_size,
                     resolution[i] - self.shift_size),
               slice(resolution[i] - self.shift_size, None))
              for i in range(len(resolution))]):
            img_mask[:, h, w, :] = index
            index += 1
        img_shape = [1]
        for size in resolution:
            img_shape += [size // self.window_size, self.window_size]
        img_mask = img_mask.reshape(img_shape)
        img_mask = img_mask.transpose((0, 1, 3, 2, 4))
        img_mask = img_mask.reshape((-1, self.window_size ** 2))
        mask = np.expand_dims(img_mask, 1) - np.expand_dims(img_mask, 2)
        mask[mask != 0] = -100.0
        mask = np.expand_dims(mask, (0, 2))
        return torch.from_numpy(mask)

    def forward(self, x, mask=None):
        if self.downsample is not None:
            x = self.downsample(x)
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0 and mask is not None:
            x = x.roll((-self.shift_size,) * 2, dims=(1, 2))
        x = space_to_depth(x, self.window_size)
        msa_shape = (-1, self.window_size ** 2, self.dim)
        wmsa_shape = (-1,) + x.shape[1:-1] + (self.window_size ** 2 * self.dim,)
        x = self.attn(x.reshape_(msa_shape), mask)
        x = depth_to_space(x.reshape_(wmsa_shape), self.window_size)
        if self.shift_size > 0 and mask is not None:
            x = x.roll((self.shift_size,) * 2, dims=(1, 2))
        x = self.drop_path(x).add_(shortcut)
        return self.drop_path(self.mlp(self.norm2(x))).add_(x)


class SwinTransformer(nn.Module):
    """SwinTransformer."""

    def __init__(self, depths, dims, num_heads, mlp_ratios,
                 patch_size=4, window_size=7, num_classes=1000, drop_path=0):
        super(SwinTransformer, self).__init__()
        drop_path = (torch.linspace(
            0, drop_path, sum(depths), dtype=torch.float32).tolist()
            if drop_path > 0 else [drop_path] * sum(depths))
        self.patch_embed = PatchEmbed(dims[0], patch_size)
        self.blocks = nn.ModuleList()
        for i, depth in enumerate(depths):
            downsample = PatchMerging(dims[i - 1], dims[i]) if i > 0 else None
            self.blocks += [Block(
                dim=dims[i], num_heads=num_heads[i],
                window_size=window_size,
                shift_size=(0 if j % 2 == 0 else window_size // 2),
                mlp_ratio=mlp_ratios[i], qkv_bias=True,
                drop_path=drop_path[len(self.blocks) - 1],
                downsample=downsample if j == 0 else None)
                for j in range(depth)]
        self.masks = dict()
        self.norm = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier = nn.Linear if num_classes > 0 else nn.Identity
        self.fc = classifier(dims[-1], num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            resolution, mask = list(x.shape[1:-1]), None
            if blk.shift_size > 0 and min(resolution) > blk.window_size:
                mask = self.masks.get(str(resolution), None)
                if mask is None:
                    mask = blk.get_mask(resolution)
                    self.masks[str(resolution)] = mask
                mask = mask.to(x)
            x = blk(x)
        x = self.norm(x)
        if x.device.type != 'mlu':
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.avgpool(x).flatten_(1))


def swin_tiny_patch4_window7_224(num_classes=1000):
    return SwinTransformer(depths=(2, 2, 6, 2), dims=(96, 192, 384, 768),
                           num_heads=(3, 6, 12, 24), mlp_ratios=(4, 4, 4, 4),
                           patch_size=4, window_size=7, drop_path=0.2,
                           num_classes=num_classes)


def swin_small_patch4_window7_224(num_classes=1000):
    return SwinTransformer(depths=(2, 2, 18, 2), dims=(96, 192, 384, 768),
                           num_heads=(3, 6, 12, 24), mlp_ratios=(4, 4, 4, 4),
                           patch_size=4, window_size=7, drop_path=0.3,
                           num_classes=num_classes)


def swin_base_patch4_window7_224(num_classes=1000):
    return SwinTransformer(depths=(2, 2, 18, 2), dims=(128, 256, 512, 1024),
                           num_heads=(4, 8, 16, 32), mlp_ratios=(4, 4, 4, 4),
                           patch_size=4, window_size=7, drop_path=0.5,
                           num_classes=num_classes)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:\n' + str(args))
    if torch.backends.mps.is_available():
        args.device = torch.device('mps', args.device)
    elif torch.cuda.is_available():
        args.device = torch.device('cuda', args.device)
    elif torch.mlu.is_available():
        args.device = torch.device('mlu', args.device)
    else:
        args.device = torch.device('cpu', args.device)
    use_fp16 = args.precision.lower() == 'float16'
    m = globals()[args.model]().to(device=args.device)
    m = m if args.train else m.eval()
    m = m.half() if use_fp16 else m
    criterion = nn.CrossEntropyLoss(reduction='mean')
    input = torch.zeros(args.batch_size, 3, 224, 224,
                        dtype=torch.float16 if use_fp16 else torch.float32)
    input = input.permute(0, 2, 3, 1) if args.device.type == 'mlu' else input
    input = input.to(device=args.device)
    target = torch.zeros(input.size(0), dtype=torch.int32).to(device=args.device)
    sync_t = torch.ones(1).to(device=args.device).add_(1).cpu()
    for iter in range(5):
        tic = time.time()
        with torch.enable_grad() if args.train else torch.no_grad():
            for i in range(30):
                x = m(input)
                if args.train:
                    loss = criterion(x.float(), target)
                    loss.backward()
                sync_t = sync_t.to(device=args.device).add_(1).cpu()
        diff_time = time.time() - tic
        print({'iter': iter,
               'throughout': round(30.0 / diff_time * input.size(0), 2),
               'time': round(diff_time, 3)})
