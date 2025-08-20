#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

// 全局参数 (需在你的工程中定义实际大小)
#ifndef FEATURE_DIM
#define FEATURE_DIM 64
#endif

#ifndef NUM_CLASSES
#define NUM_CLASSES 3
#endif

#ifndef SEQ_LEN 
#define SEQ_LEN 10 
#endif
extern float classifier_weights[FEATURE_DIM * NUM_CLASSES];
extern float classifier_bias[NUM_CLASSES];

// 编译期校验（C11 _Static_assert 或 C++ static_assert）
_Static_assert(FEATURE_DIM > 0 && NUM_CLASSES > 0, "dims must be positive");
_Static_assert(sizeof(float) == 4, "float must be 32-bit");
/**
 * @brief 从 buffer 更新 classifier (仅内存，不写入 NVS)
 */
void set_classifier_from_buffer(const uint8_t* buf, size_t len);

/**
 * @brief 从二进制 buffer 更新 classifier 并写入 NVS
 * 
 * @param data  输入的二进制数据
 * @param len   数据长度 (字节)
 * @return int  0 表示成功，-1 表示失败
 */
int update_classifier_from_bin(const uint8_t* data, size_t len);

/**
 * @brief 从 NVS 恢复 classifier 参数 (仅内存，不写回 NVS)
 */
void restore_classifier_from_nvs(void);

  void safe_nvs_operation() ;
