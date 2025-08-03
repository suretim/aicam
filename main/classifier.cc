#include "classifier.h"
#include <string.h>
#include <math.h>

extern float g_weights[CLASSIFIER_OUTPUT_DIM][CLASSIFIER_INPUT_DIM];
extern float g_bias[CLASSIFIER_OUTPUT_DIM];

//     float * g_weights;  // 长度为 input_dim * output_dim
//     float *g_bias;
// static int g_input_dim;
// static int g_output_dim;

// void classifier_set_params(  float * weights,   float *bias, int input_dim, int output_dim) {
//     g_weights = weights;
//     g_bias = bias;
//     g_input_dim = input_dim;
//     g_output_dim = output_dim;
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

int classifier_predict(const float *features) {
    float logits[CLASSIFIER_OUTPUT_DIM] = {0};
    for (int i = 0; i < CLASSIFIER_OUTPUT_DIM; ++i) {
        for (int j = 0; j < CLASSIFIER_INPUT_DIM; ++j) {
            logits[i] += g_weights[i][j] * features[j];
        }
        logits[i] += g_bias[i];
    }

    int predicted_class = 0;
    softmax(logits, CLASSIFIER_OUTPUT_DIM, &predicted_class);
    return predicted_class;
}
