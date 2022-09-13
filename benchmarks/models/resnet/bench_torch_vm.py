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
"""Bench ResNet."""

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
    parser.add_argument('--model', default='resnet50', help='compute model')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
    return parser.parse_args()


class BasicBlock(nn.Module):
    """Basic resnet block."""

    expansion = 1

    def __init__(self, dim_in, dim, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.relu(x.add_(shortcut))


class Bottleneck(nn.Module):
    """Bottleneck resnet block."""

    expansion = 4
    groups, width_per_group = 1, 64

    def __init__(self, dim_in, dim, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        width = int(dim * (self.width_per_group / 64.)) * self.groups
        self.conv1 = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(width, dim * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.relu(x.add_(shortcut))


class ResNet(nn.Module):
    """ResNet."""

    def __init__(self, block, depths, num_classes=1000):
        super(ResNet, self).__init__()
        dim_in, stage_dims, blocks = 64, [64, 128, 256, 512], []
        self.num_features = stage_dims[-1] * block.expansion
        self.conv1 = nn.Conv2d(3, stage_dims[0], kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stage_dims[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Blocks.
        for i, depth, dim in zip(range(4), depths, stage_dims):
            stride = 1 if i == 0 else 2
            downsample = None
            if stride != 1 or dim_in != dim * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim * block.expansion, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(dim * block.expansion))
            blocks.append(block(dim_in, dim, stride, downsample))
            dim_in = dim * block.expansion
            for _ in range(depth - 1):
                blocks.append(block(dim_in, dim))
            setattr(self, 'layer%d' % (i + 1), nn.Sequential(*blocks[-depth:]))
        self.blocks = blocks
        # Head.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier = nn.Linear if num_classes > 0 else nn.Identity
        self.fc = classifier(self.num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.avgpool(x).flatten_(1))


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


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
    criterion = nn.CrossEntropyLoss(reduction='mean')
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
