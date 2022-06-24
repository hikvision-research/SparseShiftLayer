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
from .basicunits import IBSSL, ResIBSSL, IBPool
from .FEBlock import FEBlock3n2s, FEBlock3n1s, FEBlock4n2s, FEBlock4n1s

class FENet(nn.Module):
    def __init__(self, reduction=1, expansion=1, num_classes=1000):
        super(FENet, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.in_planes = int(16 * self.reduction)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.ibssl = IBSSL(self.in_planes, self.in_planes, stride=1, expansion=4)
        self.ibpool = IBPool(self.in_planes, self.in_planes * 2, stride=1, expansion=5)
        self.feblock1 = FEBlock3n2s(self.in_planes * 2, self.in_planes * 4)
        self.feblock2 = FEBlock4n2s(self.in_planes * 4, self.in_planes * 8)
        self.feblock3 = FEBlock4n1s(self.in_planes * 8, self.in_planes * 8)
        self.feblock4 = FEBlock4n2s(self.in_planes * 8, self.in_planes * 16)
        self.feblock5 = FEBlock3n1s(self.in_planes * 16, self.in_planes * 16)
        self.conv2 = nn.Conv2d(self.in_planes * 16, 1932, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1932)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Conv2d(1932, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.ibssl(x)
        x = self.ibpool(x)
        x = self.feblock1(x)
        x = self.feblock2(x)
        x = self.feblock3(x)
        x = self.feblock4(x)
        x = self.feblock5(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x