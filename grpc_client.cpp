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
#include "sensor_module.h"  // 自定义传感器模块（相机、湿度、光感器）
#include "esp_camera.h" 
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
#define MODEL_PATH "/spiffs/model.tflite" // 你的 TFLite 模型路径
//#define INPUT_SIZE 64 // 假设输入图像是 64x64
#define MODEL_INPUT_SIZE 64

static const char *TAG = "PlantHealth_Client"; 
httpd_handle_t start_webserver();

float read_temperature_sensor(void)
{
    return 28.0;
}

float read_humidity_sensor(void)
{
    return 28.0;
}

float read_light_sensor(void)
{
    return 28.0;
}
uint8_t img_0[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE*3];

void capture_image(uint8_t  *img )
{
    //memcpy(img, img_0, MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 * sizeof(float));
 
    return  ;
}

// 传感器数据采集函数
sensor_data_t get_sensor_data() {
    sensor_data_t data;
    // 获取湿度数据
    data.temperature = read_temperature_sensor();
    data.humidity = read_humidity_sensor();
    // 获取光感器数据
    data.light_lux = read_light_sensor();
    // 获取图像数据
    capture_image( data.camera_frame); // 从相机模块获取 64x64 的图像数据
    return data;
} 
// MQTT 上传推理结果和模型更新
void upload_results(float prediction, float uncertainty) {
    
    //esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    //esp_mqtt_client_start(client);
    
    char payload[128];
    snprintf(payload, sizeof(payload), "{\"prediction\": %f, \"uncertainty\": %f}", prediction, uncertainty);
    
    //esp_mqtt_client_publish(client, MQTT_TOPIC, payload, 0, 1, 0);
    ESP_LOGI(TAG, "Result uploaded: %s", payload);
}

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
    init_camera(); 
    start_webserver(); 
    //while (1) {
        // 获取传感器数据
         sensor_data_t data = get_sensor_data();

        // 运行推理
        //run_encoder_inference(data);
        // 上传结果
        //upload_results(prediction, uncertainty);

    //    vTaskDelay(pdMS_TO_TICKS(10000)); // 每 10 秒进行一次采集和推理
    //}
 
    // 2. 运行 encoder 模型，获取特征向量
    float features[64];
    run_encoder_inference(/*input_image=*/NULL, features);  // 传入图像或模拟图像数据

    // 3. 分类预测
    int predicted = classifier_predict(features);
    printf("Predicted class: %d\n", predicted);


}
