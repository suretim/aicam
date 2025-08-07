#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_spi_flash.h"
#include "esp_wifi.h"
#include "esp_http_server.h"
#include <nvs_flash.h>
#include "mqtt_client.h"
#include <esp_task_wdt.h>

#include "img_inference.h"
#include "config_mqtt.h"
#include "config_camera.h"
#include "config_wifi.h"  
#include "classifier.h"
 
#define MQTT_TOPIC "plant_health"

// TFLite Micro 模型配置
//#define MODEL_PATH "/spiffs/model.tflite" // 你的 TFLite 模型路径
//#define INPUT_SIZE 64 // 假设输入图像是 64x64
//#define MODEL_INPUT_SIZE 64

static const char *TAG = "PlantHealth_Client"; 
httpd_handle_t start_esp32ap_webserver();

// 主循环
extern "C" void app_main(void) {
    
  esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
  
    // init_tflite_model();
    wifi_init_apsta();
    init_esp32_camera(); 
    start_esp32ap_webserver();  
    start_mqtt_client();
    tensor_run();
  //xTaskCreate(tensor_task, "tensor_task", 4096, NULL, 5, NULL);

}
