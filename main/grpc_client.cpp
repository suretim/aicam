#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_spi_flash.h"
#include "esp_wifi.h"
#include "mqtt_client.h"
#include "mqtt_upload.h"
#include "run_inference.h" 
//#include "sensor_module.h"  // 自定义传感器模块（相机、湿度、光感器）
//#include "esp_camera.h" 
#include <esp_task_wdt.h>

#include "camera_config.h"
#include "wifi_config.h"
#include "mqtt_upload.h"
#include "esp_http_server.h" 

#include "classifier.h"
// MQTT 配置
//#define MQTT_BROKER_URI "mqtt://your_broker_address"
#define MQTT_TOPIC "plant_health"

// TFLite Micro 模型配置
//#define MODEL_PATH "/spiffs/model.tflite" // 你的 TFLite 模型路径
//#define INPUT_SIZE 64 // 假设输入图像是 64x64
#define MODEL_INPUT_SIZE 64

static const char *TAG = "PlantHealth_Client"; 
httpd_handle_t start_esp32ap_webserver();

// 主循环
extern "C" void app_main(void) {
    
  
    ESP_LOGI(TAG, "ESP32 Plant Health Client Started");

    // 初始化 Wi-Fi 和 MQTT
    // esp_wifi_init();
    // esp_mqtt_client_config_t mqtt_cfg = {
    //     .uri = MQTT_BROKER_URI,
    // };
    // init_tflite_model();
    wifi_init_apsta();
    start_mqtt_client();
    //init_camera(); 
    start_esp32ap_webserver();  
    xTaskCreate(tensor_task, "tensor_task", 4096, NULL, 5, NULL);

}
