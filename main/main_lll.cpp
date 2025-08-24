// main.c (ESP-IDF project single-file illustrative)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_system.h"
//#include "nvs_flash.h"
//#include "nvs.h" 
 #include "config_mqtt.h" 
#include "config_wifi.h"  
#include <sstream>
#include "esp_log.h"
#include <esp_task_wdt.h>
#include "classifier_storage.h"


static const char *TAG = "MAIN_LLL";
 
 

// Dense forward: logits = bias + features @ weights
// features: float features[FEATURE_DIM]
// logits_out: float logits[NUM_CLASSES]
// void dense_forward(const float* features, float* logits_out) {
//     for (int c = 0; c < NUM_CLASSES; c++) {
//         float s = classifier_bias[c];
//         for (int f = 0; f < FEATURE_DIM; f++) {
//             s += features[f] * classifier_weights[f * NUM_CLASSES + c];
//         }
//         logits_out[c] = s;
//     }
// }

// softmax (inplace or output)
// void softmax(const float* logits, float* probs_out) {
//     float maxv = logits[0];
//     for (int i = 1; i < NUM_CLASSES; i++) if (logits[i] > maxv) maxv = logits[i];
//     float sum = 0.0f;
//     for (int i = 0; i < NUM_CLASSES; i++) {
//         probs_out[i] = expf(logits[i] - maxv);
//         sum += probs_out[i];
//     }
//     for (int i = 0; i < NUM_CLASSES; i++) probs_out[i] /= sum;
// }
 
 
  // Example MQTT callback (pseudo): receives classifier_weights.bin as payload
// In real code wire up esp-mqtt and call this when message arrives

void periodic_task(void *pvParameter) {
    while (1) {
        publish_feature_vector(0,1);
        vTaskDelay(pdMS_TO_TICKS(60000)); // 延遲 60 秒
    }
}
//extern void start_mqtt_client(void);  
extern void lll_tensor_run(void );  
// Example main demonstrating flow
extern "C" void app_main(void) {
 
 
 //initialize_nvs();
//    esp_err_t ret = nvs_flash_init();
//     if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
//         ESP_ERROR_CHECK(nvs_flash_erase());
//         ret = nvs_flash_init();
//     }
//     ESP_ERROR_CHECK(ret);
    init_classifier_from_header();

    initialize_nvs_robust();
    wifi_init_apsta();   
    start_mqtt_client(); 
    // 初始化 SPIFFS
    xTaskCreate(&periodic_task, "periodic_task", 8192, NULL, 5, NULL);
    // init tflite
    lll_tensor_run();
 
   
}
