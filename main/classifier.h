#pragma once
 

#define CLASSIFIER_INPUT_DIM 64
#define CLASSIFIER_OUTPUT_DIM 3
#define FISHER_LAYER 12
//void classifier_set_params(const float *weights, const float *bias, int input_dim, int output_dim);
int classifier_predict(const float *features);  // 返回预测类别索引
void update_classifier_weights_bias(const float* values, int value_count,int type) ;
void update_fishermatrix_theta(const float* values, int value_count,int type) ;