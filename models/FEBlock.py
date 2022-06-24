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


class FEBlock3n2s(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expansion=6):
        super(FEBlock3n2s, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.G = int(in_planes/4)
        self.resibssl_1 = ResIBSSL(self.G, self.G, stride=1, expansion=expansion)
        self.resibssl_2 = ResIBSSL(self.G*2, self.G*2, stride=1, expansion=expansion)
        self.ibpool = IBPool(self.in_planes, self.out_planes, stride=1, expansion=expansion)

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
        G1, G2, G3 = x[:, :self.G*2, :, :], x[:, self.G*2:self.G * 3, :, :], x[:, self.G * 3:, :, :]
        x = self.resibssl_1(G3)
        x = torch.cat([G2, x], dim=1)
        x = self.resibssl_2(x)
        x = torch.cat([G1, x], dim=1)
        x = self.ibpool(x)
        return x

class FEBlock3n1s(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expansion=6):
        super(FEBlock3n1s, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.G = int(in_planes/4)
        self.resibssl_1 = ResIBSSL(self.G, self.G, stride=1, expansion=expansion)
        self.resibssl_2 = ResIBSSL(self.G*2, self.G*2, stride=1, expansion=expansion)
        self.ibssl = IBSSL(self.in_planes, self.out_planes, stride=1, expansion=expansion)

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
        G1, G2, G3 = x[:, :self.G*2, :, :], x[:, self.G*2:self.G * 3, :, :], x[:, self.G * 3:, :, :]
        x = self.resibssl_1(G3)
        x = torch.cat([G2, x], dim=1)
        x = self.resibssl_2(x)
        x = torch.cat([G1, x], dim=1)
        x = self.ibssl(x)
        return x

class FEBlock4n2s(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expansion=6):
        super(FEBlock4n2s, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.G = int(in_planes/8)
        self.resibssl_1 = ResIBSSL(self.G, self.G, stride=1, expansion=expansion)
        self.resibssl_2 = ResIBSSL(self.G*2, self.G*2, stride=1, expansion=expansion)
        self.resibssl_3 = ResIBSSL(self.G * 4, self.G * 4, stride=1, expansion=expansion)
        self.ibpool = IBPool(self.in_planes, self.out_planes, stride=1, expansion=expansion)

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
        G1, G2 = x[:, :self.G * 4, :, :], x[:, self.G * 4:self.G * 6, :, :]
        G3, G4 = x[:, self.G * 6:self.G * 7, :, :], x[:, self.G * 7:, :, :]
        x = self.resibssl_1(G4)
        x = torch.cat([G3, x], dim=1)
        x = self.resibssl_2(x)
        x = torch.cat([G2, x], dim=1)
        x = self.resibssl_3(x)
        x = torch.cat([G1, x], dim=1)
        x = self.ibpool(x)
        return x
#
class FEBlock4n1s(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expansion=6):
        super(FEBlock4n1s, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.G = int(in_planes/8)
        self.resibssl_1 = ResIBSSL(self.G, self.G, stride=1, expansion=expansion)
        self.resibssl_2 = ResIBSSL(self.G*2, self.G*2, stride=1, expansion=expansion)
        self.resibssl_3 = ResIBSSL(self.G * 4, self.G * 4, stride=1, expansion=expansion)
        self.ibssl = IBSSL(self.in_planes, self.out_planes, stride=1, expansion=expansion)

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
        G1, G2 = x[:, :self.G * 4, :, :], x[:, self.G * 4:self.G * 6, :, :]
        G3, G4 = x[:, self.G * 6:self.G * 7, :, :], x[:, self.G * 7:, :, :]
        x = self.resibssl_1(G4)
        x = torch.cat([G3, x], dim=1)
        x = self.resibssl_2(x)
        x = torch.cat([G2, x], dim=1)
        x = self.resibssl_3(x)
        x = torch.cat([G1, x], dim=1)
        x = self.ibssl(x)
        return x
