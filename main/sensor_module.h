#ifndef SENSOR_MODULE_H
#define SENSOR_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h> 

// 传感器数据结构
typedef struct {
    // 相机数据
    int camera_width;
    int camera_height;
     uint8_t *camera_frame;  // 指向图像数据的指针，数据格式依项目定义
    //esp_image_metadata_t *camera_frame;
    // 温湿度数据
    float temperature ;  // 摄氏度
    float humidity;     // 相对湿度百分比

    // 光照数据
    float light_lux;            // 光照强度，单位lux
} sensor_data_t;

// 初始化传感器模块（相机 + 温湿度 + 光照）
bool sensor_module_init(void);

// 读取所有传感器数据，写入 sensor_data 结构体
bool sensor_module_read(sensor_data_t *data);

// 释放传感器模块资源
void sensor_module_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // SENSOR_MODULE_H
