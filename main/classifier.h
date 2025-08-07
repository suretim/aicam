#pragma once
 

//void classifier_set_params(const float *weights, const float *bias, int input_dim, int output_dim);
int classifier_predict(const float *features);  // 返回预测类别索引
void update_classifier_bias(const float* values, int value_count) ;
void update_classifier_weights(const float* values, int value_count) ;