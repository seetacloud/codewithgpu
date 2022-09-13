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
"""Bench MobileNetV3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import time

from dragon.vm import torch
from dragon.vm.torch import nn


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='run training or inference')
    parser.add_argument('--precision', default='float16', help='compute precision')
    parser.add_argument('--device', default=0, type=int, help='compute device')
    parser.add_argument('--model', default='mobilenet_v3_large', help='compute model')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    return parser.parse_args()


def make_divisible(v, divisor=8):
    """Return the divisible value."""
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNorm2d(nn.Sequential):
    """2d convolution followed by norm."""

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        norm_type='BatchNorm2d',
        activation_type='',
        inplace=True,
    ):
        super(ConvNorm2d, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [nn.Conv2d(dim_in, dim_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias and (not norm_type))]
        if norm_type:
            layers += [getattr(nn, norm_type)(dim_out)]
        if activation_type:
            layers += [getattr(nn, activation_type)()]
            layers[-1].inplace = inplace
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, dim_in, dim):
        super(SqueezeExcite, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim_in, 1)
        self.activation1 = nn.ReLU(True)
        self.activation2 = nn.Hardsigmoid(True)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.activation1(self.conv1(scale))
        scale = self.activation2(self.conv2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """Invert residual block."""

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size=3,
        stride=1,
        expand_ratio=3,
        squeeze_ratio=1,
        activation_type='ReLU',
    ):
        super(InvertedResidual, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, activation_type=activation_type)
        self.apply_shortcut = stride == 1 and dim_in == dim_out
        self.dim = dim = int(round(dim_in * expand_ratio))
        self.conv1 = (conv_module(dim_in, dim, 1)
                      if expand_ratio > 1 else nn.Identity())
        self.conv2 = conv_module(dim, dim, kernel_size, stride, groups=dim)
        self.se = (SqueezeExcite(dim, make_divisible(dim * squeeze_ratio))
                   if squeeze_ratio < 1 else nn.Identity())
        self.conv3 = conv_module(dim, dim_out, 1, activation_type='')

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.apply_shortcut:
            return x.add_(shortcut)
        return x


class MobileNetV3(nn.Module):
    """MobileNetV3 class."""

    def __init__(self, depths, dims, kernel_sizes, strides,
                 expand_ratios, squeeze_ratios, width_mult=1.0,
                 dropout=0.2, num_classes=1000):
        super(MobileNetV3, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, activation_type='Hardswish')
        dims = list(map(lambda x: make_divisible(x * width_mult), dims))
        self.conv1 = conv_module(3, dims[0], 3, 2)
        dim_in, blocks, coarsest_stride = dims[0], [], 2
        for i, (depth, dim) in enumerate(zip(depths, dims[1:])):
            coarsest_stride *= strides[i]
            layer_expand_ratios = expand_ratios[i]
            if not isinstance(layer_expand_ratios, (tuple, list)):
                layer_expand_ratios = [layer_expand_ratios]
            layer_expand_ratios = list(layer_expand_ratios)
            layer_expand_ratios += ([layer_expand_ratios[-1]] *
                                    (depth - len(layer_expand_ratios)))
            for j in range(depth):
                blocks.append(InvertedResidual(
                    dim_in, dim,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i] if j == 0 else 1,
                    expand_ratio=layer_expand_ratios[j],
                    squeeze_ratio=squeeze_ratios[i],
                    activation_type='Hardswish'
                    if coarsest_stride >= 16 else 'ReLU'))
                dim_in = dim
            setattr(self, 'layer%d' % (i + 1), nn.Sequential(*blocks[-depth:]))
        self.conv2 = conv_module(dim_in, blocks[-1].dim, 1)
        self.blocks = blocks + [self.conv2]
        # Head.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(blocks[-1].dim, dims[-1]),
            nn.Hardswish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(dims[-1], num_classes),
        ) if num_classes > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.avgpool(x).flatten_(1))


def mobilenet_v3_large(num_classes=1000):
    return MobileNetV3(
        dims=(16,) + (16, 24, 40, 80, 112, 160) + (1280,),
        depths=(1, 2, 3, 4, 2, 3),
        kernel_sizes=(3, 3, 5, 3, 3, 5),
        strides=(1, 2, 2, 2, 1, 2),
        expand_ratios=(1, (4, 3), 3, (6, 2.5, 2.3, 2.3), 6, 6),
        squeeze_ratios=(1, 1, 0.25, 1, 0.25, 0.25),
        num_classes=num_classes)


def mobilenet_v3_small(num_classes=1000):
    return MobileNetV3(
        dims=(16,) + (16, 24, 40, 48, 96) + (1024,),
        depths=(1, 2, 3, 2, 3),
        kernel_sizes=(3, 3, 5, 5, 5),
        strides=(2, 2, 2, 1, 2),
        expand_ratios=(1, (4.5, 88. / 24), (4, 6, 6), 3, 6),
        squeeze_ratios=(0.25, 1, 0.25, 0.25, 0.25),
        num_classes=num_classes)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:\n' + str(args))
    if torch.backends.mps.is_available():
        args.device = torch.device('mps', args.device)
    elif torch.cuda.is_available():
        args.device = torch.device('cuda', args.device)
    else:
        args.device = torch.device('cpu', args.device)
    use_fp16 = args.precision.lower() == 'float16'
    m = globals()[args.model]().to(device=args.device)
    m = m if args.train else m.eval()
    m = m.half() if use_fp16 else m
    criterion = nn.CrossEntropyLoss()
    input = torch.zeros(args.batch_size, 3, 224, 224,
                        dtype=torch.float16 if use_fp16 else torch.float32)
    input = input.to(device=args.device)
    target = torch.zeros(input.size(0), dtype=torch.int64).to(device=args.device)
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
