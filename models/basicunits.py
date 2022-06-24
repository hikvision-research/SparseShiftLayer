# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .SSL import SSL2d

class IBSSL(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(IBSSL, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        # self.mid_planes = mid_planes = int(in_planes * self.expansion)
        self.mid_planes = mid_planes = int(out_planes * self.expansion)
        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shift2 = SSL2d(mid_planes)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(self.shift2(x)))
        self.out_nchw = x.size()
        return x

class ResIBSSL(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(ResIBSSL, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        # self.mid_planes = mid_planes = int(in_planes * self.expansion)
        self.mid_planes = mid_planes = int(out_planes * self.expansion)
        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shift2 = SSL2d(mid_planes)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)


    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = x.size()
        x = self.bn2(self.conv2(self.shift2(x)))
        x += shortcut
        self.out_nchw = x.size()
        return x

class IBPool(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(IBPool, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        # self.mid_planes = mid_planes = int(in_planes * self.expansion)
        self.mid_planes = mid_planes = int(out_planes * self.expansion)

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shift2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = x.size()
        x = self.bn2(self.conv2(self.shift2(x)))
        self.out_nchw = x.size()
        return x