// Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

__global__ void ssl_cuda_kernel(const float* input,
                            float* output,
                            int batch_size,
                            int channels,
                            int height,
                            int width,
                            float* a,
                            float* b,
                            int num) {
    int spatial_size = height * width;
    int dim = channels * spatial_size;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < dim*batch_size; index += gridDim.x * blockDim.x) {
        int n = index / dim;
        int c = index % dim / spatial_size;
        int h = index % spatial_size / width;
        int w = index % width;
        h += round(a[c]);
        w += round(b[c]);
        bool state = (h >= 0 && h < height && w >= 0 && w < width);
        output[index] = state * input[state * (w + h * width + c * spatial_size + n * dim)];
    }
}

void ssl_cuda(const float* input,
                 float* output,
                 int batch_size,
                 int channels,
                 int height,
                 int width,
                 float* a,
                 float* b,
                 int num) {
    dim3 grid((num + 1023) / 1024);
    dim3 block(1024);
    ssl_cuda_kernel<<<grid, block>>>(input, output, batch_size, channels, height, width, a, b, num);
}

__global__ void ssl_diff_cuda_kernel(const float* grad_output,
                            float* grad_intput,
                            int batch_size,
                            int channels,
                            int height,
                            int width,
                            float* a,
                            float* b,
                            int num) {
    int spatial_size = height * width;
    int dim = channels * spatial_size;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < dim*batch_size; index += gridDim.x * blockDim.x) {
    int n = index / dim;
    int c = index % dim / spatial_size;
    int h = index % spatial_size / width;
    int w = index % width;
    h -= round(a[c]);
    w -= round(b[c]);
    bool state = (h >= 0 && h < height && w >= 0 && w < width);
    grad_intput[index] = state * grad_output[state * (w + h * width + c * spatial_size + n * dim)];
  }
}

void ssl_diff_cuda(const float* grad_output,
                 float* grad_intput,
                 int batch_size,
                 int channels,
                 int height,
                 int width,
                 float* a,
                 float* b,
                 int num) {
    dim3 grid((num + 1023) / 1024);
    dim3 block(1024);
    ssl_diff_cuda_kernel<<<grid, block>>>(grad_output, grad_intput, batch_size, channels, height, width, a, b, num);
}

__global__ void ssl_off_diff_cuda_kernel(const float* grad_output,
                            const float* intput,
                            int batch_size,
                            int channels,
                            int height,
                            int width,
                            float* a,
                            float* b,
                            float* I_a,
                            float* I_b,
                            int num) {
    int spatial_size = height * width;
    int dim = channels * spatial_size;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
            index < dim*batch_size; index += gridDim.x * blockDim.x) {
    int n = index / dim;
    int c = index % dim / spatial_size;
    int idx = index % spatial_size;
    int h = idx / width;
    int w = idx % width;
    int dst_h = h + a[c];
    int dst_w = w + b[c];

    float h_weight_r = h + a[c] - dst_h;
    float h_weight_l = 1 - h_weight_r;
    float w_weight_r = w + b[c] - dst_w;
    float w_weight_l = 1 - w_weight_r;

    bool state1 = dst_h >= 0       && dst_h < height       && dst_w >= 0       && dst_w < width;
    bool state2 = (dst_h + 1) >= 0 && (dst_h + 1) < height && dst_w >= 0       && dst_w < width;
    bool state3 = dst_h >= 0       && dst_h < height       && (dst_w + 1) >= 0 && (dst_w + 1) < width;
    bool state4 = (dst_h + 1) >= 0 && (dst_h + 1) < height && (dst_w + 1) >= 0 && (dst_w + 1) < width;

    int bottom_offset = c * spatial_size + n * dim;

    //w.r.t. h_offset
    float Ia = state1 * intput[(bottom_offset + dst_h * width + dst_w) * state1] * w_weight_l * (-1.0)+
                         state2 * intput[(bottom_offset + (dst_h + 1) * width + dst_w) * state2] * w_weight_l +
                         state3 * intput[(bottom_offset + dst_h * width + dst_w + 1) * state3] * w_weight_r * (-1.0) +
                         state4 * intput[(bottom_offset + (dst_h + 1) * width + dst_w + 1) * state4] * w_weight_r;
    I_a[index] = Ia * grad_output[index];
    //w.r.t. w_offset
    float Ib = state1 * intput[(bottom_offset + dst_h * width + dst_w) * state1] * h_weight_l * (-1.0)+
                         state2 * intput[(bottom_offset + (dst_h + 1) * width + dst_w) * state2] * h_weight_r * (-1.0) +
                         state3 * intput[(bottom_offset + dst_h * width + dst_w + 1) * state3] * h_weight_l +
                         state4 * intput[(bottom_offset + (dst_h + 1) * width + dst_w + 1) * state4] * h_weight_r;
    I_b[index] = Ib * grad_output[index];
  }
}

void ssl_off_diff_cuda(const float* grad_output,
                 float* intput,
                 int batch_size,
                 int channels,
                 int height,
                 int width,
                 float* a,
                 float* b,
                 float* I_a,
                 float* I_b,
                 int num) {
    dim3 grid((num + 1023) / 1024);
    dim3 block(1024);
    ssl_off_diff_cuda_kernel<<<grid, block>>>(grad_output, intput, batch_size, channels, height, width, a, b, I_a, I_b, num);
}