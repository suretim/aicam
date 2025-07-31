#pragma once

#include <stdint.h>

#define CLASSIFIER_INPUT_DIM 64
#define CLASSIFIER_OUTPUT_DIM 2

void classifier_set_params(const float *weights, const float *bias, int input_dim, int output_dim);
int classifier_predict(const float *features);  // 返回预测类别索引
