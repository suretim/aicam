#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 输入图像：宽高 64x64，RGB888（uint8_t[64 * 64 * 3]）
// 输出特征向量：float[64]
void run_encoder_inference(const uint8_t *input_image, float *output_vector);

#ifdef __cplusplus
}
#endif
