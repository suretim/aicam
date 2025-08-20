// main.c (ESP-IDF project single-file illustrative)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_system.h"
//#include "nvs_flash.h"
//#include "nvs.h" 
 
#include <sstream>
#include "esp_log.h"
#include <esp_task_wdt.h>
#include "classifier_storage.h"
static const char *TAG = "MAIN_LLL";
 
 

// Dense forward: logits = bias + features @ weights
// features: float features[FEATURE_DIM]
// logits_out: float logits[NUM_CLASSES]
void dense_forward(const float* features, float* logits_out) {
    for (int c = 0; c < NUM_CLASSES; c++) {
        float s = classifier_bias[c];
        for (int f = 0; f < FEATURE_DIM; f++) {
            s += features[f] * classifier_weights[f * NUM_CLASSES + c];
        }
        logits_out[c] = s;
    }
}

// softmax (inplace or output)
void softmax(const float* logits, float* probs_out) {
    float maxv = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs_out[i] = expf(logits[i] - maxv);
        sum += probs_out[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) probs_out[i] /= sum;
}
 
 
  // Example MQTT callback (pseudo): receives classifier_weights.bin as payload
// In real code wire up esp-mqtt and call this when message arrives
void mqtt_on_message(const uint8_t* payload, size_t payload_len) {
    // payload should be raw float32 bytes as produced by export script
    int rc = update_classifier_from_bin(payload, payload_len);
    if (rc == 0) {
        printf("MQTT classifier update applied.\n");
    } else {
        printf("MQTT classifier update failed.\n");
    }
}

  
extern void lll_tensor_run();

// Example main demonstrating flow
extern "C" void app_main(void) {
 
 
 //initialize_nvs();
  

    // initialize classifier default from header if compiled in
    // restore persisted classifier if any
    //restore_classifier_from_nvs();
    safe_nvs_operation() ;
    // 初始化 SPIFFS
    
    // init tflite
    lll_tensor_run();

    // Now in your app loop you would:
    // - capture image, preprocess to float array matching TFLite input
    // - call run_encoder(image, features)
    // - call dense_forward(features, logits)
    // - optionally softmax and decide class
    // For demonstration we do nothing more.
    // while (1) {
    //     vTaskDelay(1000 / portTICK_PERIOD_MS);
    // }
   
}
