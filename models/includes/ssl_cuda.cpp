/*
Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <torch/extension.h>
#include "./ssl_cuda.h"

void torch_ssl_cuda(const torch::Tensor &input,
                        torch::Tensor &output,
                        int64_t batch_size,
                        int64_t channels,
                        int64_t height,
                        int64_t width,
                        torch::Tensor &a,
                        torch::Tensor &b,
                        int64_t num) {
    ssl_cuda((const float *)input.data_ptr(),
                (float *)output.data_ptr(),
                batch_size,
                channels,
                height,
                width,
                (float *)a.data_ptr(),
                (float *)b.data_ptr(),
                num);
}

void torch_ssl_diff_cuda(const torch::Tensor &grad_output,
                        torch::Tensor &grad_intput,
                        int64_t batch_size,
                        int64_t channels,
                        int64_t height,
                        int64_t width,
                        torch::Tensor &a,
                        torch::Tensor &b,
                        int64_t num) {
    ssl_diff_cuda((const float *)grad_output.data_ptr(),
                (float *)grad_intput.data_ptr(),
                batch_size,
                channels,
                height,
                width,
                (float *)a.data_ptr(),
                (float *)b.data_ptr(),
                num);
}

void torch_ssl_off_diff_cuda(const torch::Tensor &grad_output,
                        torch::Tensor &intput,
                        int64_t batch_size,
                        int64_t channels,
                        int64_t height,
                        int64_t width,
                        torch::Tensor &a,
                        torch::Tensor &b,
                        torch::Tensor &I_a,
                        torch::Tensor &I_b,
                        int64_t num) {
    ssl_off_diff_cuda((const float *)grad_output.data_ptr(),
                (float *)intput.data_ptr(),
                batch_size,
                channels,
                height,
                width,
                (float *)a.data_ptr(),
                (float *)b.data_ptr(),
                (float *)I_a.data_ptr(),
                (float *)I_b.data_ptr(),
                num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_ssl_cuda",
          &torch_ssl_cuda,
          "ssl kernel warpper");
    m.def("torch_ssl_diff_cuda",
          &torch_ssl_diff_cuda,
          "ssl kernel warpper");
    m.def("torch_ssl_off_diff_cuda",
          &torch_ssl_off_diff_cuda,
          "ssl kernel warpper");
}
