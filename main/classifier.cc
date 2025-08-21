#include "classifier.h" 
#include "config_mqtt.h"
#include "esp_log.h" 
//#include <string>
//#include <sstream>
#include <stdlib.h> 
#include <math.h>

#include <string.h>
#include <stdio.h>  

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  


// 全局变量保存权重和偏置（可以从 protobuf 接收到的参数更新）
//float g_dense_weights[CLASSIFIER_OUTPUT_DIM][CLASSIFIER_INPUT_DIM];
//float g_dense_bias[CLASSIFIER_OUTPUT_DIM];

//float fisher_matrix[FISHER_LAYER*CLASSIFIER_INPUT_DIM];
//float theta[FISHER_LAYER*CLASSIFIER_INPUT_DIM];
 
 //float *fisher_matrix ;   // 每个变量的 Fisher 数组
 //float *theta ;    // 上一次权重
//float *theta_old ;    // 上一次权重
constexpr int FisherArenaSize = FISHER_LAYER * 256 *sizeof(float); // 调整大小
void alloc_fisher_matrix()
{
    if (fisher_matrix == NULL) {
        fisher_matrix = (float*) heap_caps_malloc(FisherArenaSize, MALLOC_CAP_SPIRAM);
    }
    if (theta == NULL) {
        theta = (float*) heap_caps_malloc(FisherArenaSize, MALLOC_CAP_SPIRAM);
    }
}

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
    publish_feature_vector(predicted_class,0);
    return predicted_class;
} 
 

// 更新分类器的权重和偏置
void update_classifier_weights_bias(const float* values, int value_count,int type) {
    
    
    int expected = CLASSIFIER_OUTPUT_DIM * CLASSIFIER_INPUT_DIM ;
    if(type==1)
    {
        expected=CLASSIFIER_OUTPUT_DIM;
    }
    
    if (value_count != expected) {
        printf("权重参数数量不匹配，期望 %d，实际 %d\n", expected, value_count);
        return;
    }
    if(type==0)
    {
        // 前 NUM_CLASSES * NUM_INPUTS 是权重
        for (int i = 0; i < CLASSIFIER_OUTPUT_DIM; ++i) {
            for (int j = 0; j < CLASSIFIER_INPUT_DIM; ++j) {
                g_dense_weights[i][j] = values[i * CLASSIFIER_INPUT_DIM + j];
            }
        }
    }
    else{    
        // 后 NUM_CLASSES 是 bias
        for (int i = 0; i < CLASSIFIER_OUTPUT_DIM; ++i) {
            g_dense_bias[i] = values[ i];
        }
    }

}
void update_fishermatrix_theta(const float* values, int value_count,int type) {
     
     
    if(type==0)
    {
        memcpy(fisher_matrix, values, value_count);
        return;
    }
    else{
        memcpy(theta, values,value_count);
        return;    
    }
     

    printf("分类器权重更新完成\n");
}
