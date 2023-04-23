# ------------------------------------------------------------------------
# Copyright (c) 2022-present, SeetaCloud, Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Bench Vision Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

from dragon.vm import torch
from dragon.vm.torch import nn


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='run training or inference')
    parser.add_argument('--precision', default='float16', help='compute precision')
    parser.add_argument('--device', default=0, type=int, help='compute device')
    parser.add_argument('--model', default='vit_base_patch16_224', help='compute model')
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size')
    return parser.parse_args()


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

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        qkv_shape = (-1, x.size(1), 3, self.num_heads, self.head_dim)
        qkv = self.qkv(x).reshape_(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0, copy=x.device.type == 'mps')
        attn = q @ k.transpose(-2, -1).mul_(self.scale)
        attn = nn.functional.softmax(attn, dim=-1, inplace=True)
        return self.proj((attn @ v).transpose(1, 2).flatten_(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, drop_path=0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path = nn.DropPath(drop_path, inplace=True)

    def forward(self, x):
        x = self.drop_path(self.attn(self.norm1(x))).add_(x)
        return self.drop_path(self.mlp(self.norm2(x))).add_(x)


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, dim=768, patch_size=16):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        if x.device.type == 'mlu':
            return x.flatten_(1, 2)
        return x.flatten_(2).transpose(1, 2)


class PosEmbed(nn.Module):
    """Position embedding layer."""

    def __init__(self, dim, num_patches):
        super(PosEmbed, self).__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.weight = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x.add_(self.weight)


class VisionTransformer(nn.Module):
    """Vision Transformer."""

    def __init__(self, depths, dims, num_heads, mlp_ratios,
                 img_size=224, patch_size=16, drop_path=0, num_classes=1000):
        super(VisionTransformer, self).__init__()
        drop_path = (torch.linspace(
            0, drop_path, sum(depths), dtype=torch.float32).tolist()
            if drop_path > 0 else [drop_path] * sum(depths))
        self.num_patches = (img_size // patch_size) ** 2
        self.num_features = dims[0]
        self.patch_embed = PatchEmbed(dims[0], patch_size)
        self.pos_embed = PosEmbed(dims[0], self.num_patches + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dims[0]))
        self.blocks = nn.ModuleList([Block(
            dim=dims[0], num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0], qkv_bias=True,
            drop_path=drop_path[i]) for i in range(depths[0])])
        self.norm = nn.LayerNorm(self.num_features)
        classifier = nn.Linear if num_classes > 0 else nn.Identity
        self.fc = classifier(self.num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gelu_approximate = 'none'
        if torch.backends.mps.is_available():
            gelu_approximate = 'tanh'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GELU):
                m.approximate = gelu_approximate
        nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.norm(x[:, 1:].mean(1)))


def vit_small_patch16_224(num_classes=1000):
    return VisionTransformer(depths=(12,), dims=(384,), num_heads=(6,),
                             mlp_ratios=(4,), img_size=224, patch_size=16,
                             drop_path=0.1, num_classes=num_classes)


def vit_medium_patch16_224(num_classes=1000):
    return VisionTransformer(depths=(16,), dims=(768,), num_heads=(12,),
                             mlp_ratios=(3,), img_size=224, patch_size=16,
                             drop_path=0.1, num_classes=num_classes)


def vit_base_patch16_224(num_classes=1000):
    return VisionTransformer(depths=(12,), dims=(768,), num_heads=(12,),
                             mlp_ratios=(4,), img_size=224, patch_size=16,
                             drop_path=0.1, num_classes=num_classes)


def vit_large_patch16_224(num_classes=1000):
    return VisionTransformer(depths=(24,), dims=(1024,), num_heads=(16,),
                             mlp_ratios=(4,), img_size=224, patch_size=16,
                             drop_path=0.1, num_classes=num_classes)


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
