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
from torch.utils.cpp_extension import load
from torch.autograd import Variable
cuda_module = load(name="ssl_cuda",
                   sources=["./models/includes/ssl_cuda.cpp",
                            "./models/includes/ssl_cuda_kernels.cu"],
                   verbose=True)

class SSLFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b) 
        batch_size, channels, height, width = x.size()
        num = batch_size * channels * height * width
        x = x.contiguous().view(-1)
        output = torch.zeros_like(x, dtype=torch.float)
        cuda_module.torch_ssl_cuda(x, output, batch_size, channels, height, width, a, b, num)
        output = output.contiguous().view(batch_size, channels, height, width)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        x, a, b = ctx.saved_variables
        batch_size, channels, height, width = x.size()
        num = batch_size * channels * height * width
        grad_output = grad_output.contiguous().view(-1)
        grad_input = torch.zeros_like(grad_output, dtype=torch.float)
        cuda_module.torch_ssl_diff_cuda(grad_output, grad_input, batch_size, channels, height, width, a, b, num)
        grad_input = grad_input.contiguous().view(batch_size, channels, height, width)


        x = x.contiguous().view(-1)
        I_a = torch.zeros_like(grad_output, dtype=torch.float)
        I_b = torch.zeros_like(grad_output, dtype=torch.float)
        cuda_module.torch_ssl_off_diff_cuda(grad_output, x, batch_size, channels, height, width, a, b, I_a, I_b, num)

        I_a = I_a.contiguous().view(batch_size, channels, height, width)
        I_b = I_b.contiguous().view(batch_size, channels, height, width)
        grad_a = torch.sum(I_a, dim=(0, 2, 3))
        grad_b = torch.sum(I_b, dim=(0, 2, 3))

        return grad_input, grad_a, grad_b 

class SSL2d(nn.Module):
    def __init__(self, inplanes):
        super(SSL2d, self).__init__()
        self.a = nn.Parameter(torch.Tensor(inplanes))
        self.b = nn.Parameter(torch.Tensor(inplanes))
        self.a.data.uniform_(0., 0.)
        self.b.data.uniform_(0., 0.)
        group_size = inplanes // 9
        main_size = group_size * 9
        for i in range(main_size):
            a_offset = i / group_size / 3. - 1.
            b_offset = i / group_size % 3. - 1.
            self.a[i].data.uniform_(a_offset, a_offset)
            self.b[i].data.uniform_(b_offset, b_offset)

    def forward(self, x):
        out = SSLFunction.apply(x, self.a, self.b)

        return out
