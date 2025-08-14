#include "classifier.h"
#include "mqtt_client.h"

#include "esp_log.h" 
#include <string>
#include <sstream>
#include <stdlib.h> 
#include <math.h>
 
#define CLASSIFIER_INPUT_DIM 64
#define CLASSIFIER_OUTPUT_DIM 3

// 全局变量保存权重和偏置（可以从 protobuf 接收到的参数更新）
float g_dense_weights[CLASSIFIER_OUTPUT_DIM][CLASSIFIER_INPUT_DIM];
float g_dense_bias[CLASSIFIER_OUTPUT_DIM];



// float g_weights[CLASSIFIER_OUTPUT_DIM][CLASSIFIER_INPUT_DIM];
// float g_bias[CLASSIFIER_OUTPUT_DIM];
// void classifier_set_params(const float *weights, const float *bias, int input_dim, int output_dim) 
// {
//     memcpy(g_weights, weights, sizeof(g_weights));
//     memcpy(g_bias, bias, sizeof(g_bias));
//     return;
// }

static float softmax(float x[], int len, int *out_index) {
    float max = x[0];
    for (int i = 1; i < len; ++i)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    float probs[len];
    for (int i = 0; i < len; ++i) {
        probs[i] = expf(x[i] - max);
        sum += probs[i];
    }

    int max_idx = 0;
    float max_prob = 0;
    for (int i = 0; i < len; ++i) {
        probs[i] /= sum;
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }

    if (out_index) *out_index = max_idx;
    return max_prob;
}
void publish_feature_vector(int label );
int classifier_predict(const float *features) {
    float logits[CLASSIFIER_OUTPUT_DIM] = {0};
    for (int i = 0; i < CLASSIFIER_OUTPUT_DIM; ++i) {
        for (int j = 0; j < CLASSIFIER_INPUT_DIM; ++j) {
            logits[i] += g_dense_weights[i][j] * features[j];
        }
        logits[i] += g_dense_bias[i];
    }

    int predicted_class = 0;
    softmax(logits, CLASSIFIER_OUTPUT_DIM, &predicted_class);
    publish_feature_vector(predicted_class);
    return predicted_class;
} 
 

// 更新分类器的权重和偏置
void update_classifier_weights(const float* values, int value_count) {
    int expected = CLASSIFIER_OUTPUT_DIM * CLASSIFIER_INPUT_DIM ;
    if (value_count != expected) {
        printf("权重参数数量不匹配，期望 %d，实际 %d\n", expected, value_count);
        return;
    }

    // 前 NUM_CLASSES * NUM_INPUTS 是权重
    for (int i = 0; i < CLASSIFIER_OUTPUT_DIM; ++i) {
        for (int j = 0; j < CLASSIFIER_INPUT_DIM; ++j) {
            g_dense_weights[i][j] = values[i * CLASSIFIER_INPUT_DIM + j];
        }
    }
}
void update_classifier_bias(const float* values, int value_count) {
    int expected =  CLASSIFIER_OUTPUT_DIM ;
    if (value_count != expected) {
        printf("权重参数数量不匹配，期望 %d，实际 %d\n", expected, value_count);
        return;
    }

    // 后 NUM_CLASSES 是 bias
    for (int i = 0; i < CLASSIFIER_OUTPUT_DIM; ++i) {
        g_dense_bias[i] = values[ i];
    }

    printf("分类器权重更新完成\n");
}
