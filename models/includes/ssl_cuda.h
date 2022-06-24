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
void ssl_cuda(const float *input,
                        float *output,
                        int batch_size,
                        int channels,
                        int height,
                        int width,
                        float *a,
                        float *b,
                        int num);
void ssl_diff_cuda(const float *grad_output,
                        float *grad_intput,
                        int batch_size,
                        int channels,
                        int height,
                        int width,
                        float *a,
                        float *b,
                        int num);
void ssl_off_diff_cuda(const float *grad_output,
                        float *intput,
                        int batch_size,
                        int channels,
                        int height,
                        int width,
                        float *a,
                        float *b,
                        float *I_a,
                        float *I_b,
                        int num);