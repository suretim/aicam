  
#include <stdio.h>
#include <math.h>
#include "esp_log.h"
#include "nvs_flash.h"
#include "camera_config.h"
#include "mqtt_upload.h"

#include "encoder_model.h"
#include "class_prototypes.h"

//#include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/c/common.h"
#include "C:\\Espressif\\frameworks\\esp-idf-v5.1.2\\examples\\esp32_project\\components\\tflite-micro\\tensorflow\\lite\\micro\\all_ops_resolver.h"
#include "C:\\Espressif\\frameworks\\esp-idf-v5.1.2\\examples\\esp32_project\\components\\tflite-micro\\tensorflow\\lite\\micro\\micro_interpreter.h"
#include "C:\\Espressif\\frameworks\\esp-idf-v5.1.2\\examples\\esp32_project\\components\\tflite-micro\\tensorflow\\lite\\schema\\schema_generated.h"
#include "C:\\Espressif\\frameworks\\esp-idf-v5.1.2\\examples\\esp32_project\\components\\tflite-micro\\tensorflow\\lite\\c\common.h"
 
#include "esp_camera.h" 
#include "mqtt_client.h" 

#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_event.h"
#define MODEL_INPUT_SIZE 64
#define TENSOR_ARENA_SIZE (40 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
// void wait_for_wifi_connection() {
//     // 实际项目应通过事件组或状态检查等待连接成功
//     vTaskDelay(pdMS_TO_TICKS(5000)); // 暂停 5 秒，模拟等待
// }

#include "freertos/event_groups.h"

#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t wifi_event_group;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wait_for_wifi_connection() {
    // 等待 WIFI_CONNECTED_BIT 设置，最长等 10 秒
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdTRUE,
                                           pdMS_TO_TICKS(10000));
    if (!(bits & WIFI_CONNECTED_BIT)) {
        ESP_LOGE("WIFI", "WiFi 连接超时！");
    } else {
        ESP_LOGI("WIFI", "WiFi 已连接！");
    }
}


void init_camera() {
    camera_config_t config = {
        .pin_pwdn  = -1,
        .pin_reset = -1,
        .pin_xclk = 10,
        .pin_sccb_sda = 8,
        .pin_sccb_scl = 7,

        .pin_d7 = 21,
        .pin_d6 = 20,
        .pin_d5 = 19,
        .pin_d4 = 18,
        .pin_d3 = 17,
        .pin_d2 = 16,
        .pin_d1 = 15,
        .pin_d0 = 14,
        .pin_vsync = 6,
        .pin_href = 5,
        .pin_pclk = 4,

        .xclk_freq_hz = 20000000,  // 20MHz，OV2640 推荐值
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        .pixel_format = PIXFORMAT_JPEG, // 或 RGB565/GRAYSCALE
        .frame_size = FRAMESIZE_QVGA,
        .jpeg_quality = 12,
        .fb_count = 1
    };


    camera_config_t config1 = {
        .pin_pwdn = -1,
        .pin_reset = -1,
        .pin_xclk = 10,
        .pin_sccb_sda = 8,
        .pin_sccb_scl = 7,
        .pin_d7 = 15,
        .pin_d6 = 17,
        .pin_d5 = 18,
        .pin_d4 = 16,
        .pin_d3 = 14,
        .pin_d2 = 12,
        .pin_d1 = 11,
        .pin_d0 = 13,
        .pin_vsync = 6,
        .pin_href = 4,
        .pin_pclk = 5,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_GRAYSCALE,
        .frame_size = FRAMESIZE_96X96,
        .jpeg_quality = 12,
        .fb_count = 1
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE("CAM", "Camera init failed: 0x%x", err);
    } else {
        ESP_LOGI("CAM", "✅ Camera ready");
    }
}


#define TAG "MQTT"

static esp_mqtt_client_handle_t client = NULL;
 
// 将返回类型改为 void
void mqtt_event_handler_cb(void* handler_arg, esp_event_base_t event_base,
                            int32_t event_id, void* event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    // 处理 MQTT 事件
    switch (event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT Connected");
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT Disconnected");
            break;
        // 处理其他事件...
    }
}
void mqtt_app_start() {
    //const esp_mqtt_client_config_t mqtt_cfg = {
    //    .broker.address.uri = "mqtt://192.168.0.57:1883"
    //};
    const esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = "mqtt://192.168.0.57:1883"
            }
        }
    };
    client = esp_mqtt_client_init(&mqtt_cfg);
    #define ESP_EVENT_ANY_ID     ((esp_mqtt_event_id_t) -1)
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler_cb, NULL);
    
    esp_mqtt_client_start(client);
}

void mqtt_send_result(const char* json) {
    if (client) {
        esp_mqtt_client_publish(client, "leaf/status", json, 0, 1, 0);
    }
}  
 

void start_mqtt_client() {
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = "mqtt://192.168.0.57:1883"
            }
        },
        .credentials = {
            .username = "tim",
            .authentication = {
                .password = "tim"
            }
        },
        .session = {
            .last_will = { 0 },  // 必须先初始化这个字段
            .disable_clean_session = false,
            .keepalive = 60
        },
        .network = {
            .reconnect_timeout_ms = 3000
        }
    };

    static char client_id[64];
    snprintf(client_id, sizeof(client_id), "smartagriculture1/leaf_detection");
    mqtt_cfg.credentials.client_id = client_id;

    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_start(client);

    // 示例发送结果
    char json[128];
    snprintf(json, sizeof(json),
             "{\"smartagriculture1\":\"%s\", \"distance\":%.3f}", class_names[0], 0.5);
    mqtt_send_result(json);
}

 extern "C" void app_main(void) {
    // 1. 初始化 NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    // 2. 初始化 TCP/IP 栈和 WiFi 驱动
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    // 配置 WiFi
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "1573",
            .password = "987654321",
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // 3. 等待连接成功（使用事件组或回调函数）
    wait_for_wifi_connection();  // 自己实现等待连接的函数
esp_netif_ip_info_t ip_info;
esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &ip_info);
ESP_LOGI(TAG, "ESP32 IP: " IPSTR, IP2STR(&ip_info.ip));

    // 4. Wi-Fi 连上后，再开始 MQTT
    start_mqtt_client();
}

// extern "C" void app_main1(void) {
//     //nvs_flash_init();
//     ESP_LOGI("MAIN", "app_main started.");
//       mqtt_app_start();

//     const tflite::Model* model = tflite::GetModel(encoder_model);
//     static tflite::AllOpsResolver resolver;
//     static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
     

//     interpreter.AllocateTensors();

//     TfLiteTensor* input = interpreter.input(0);
//     TfLiteTensor* output = interpreter.output(0);
//     //init_camera();
//      //while (1) 
//      {
//         ESP_LOGI("MAIN", "app_main loop");
// #if 0
//         camera_fb_t* fb = esp_camera_fb_get();
//         if (!fb) {
//             printf("Camera capture failed\n");
//             continue;
//         }



 
//         // 原图像尺寸（视你配置的 frame_size 决定）
//         const int src_w = fb->width;
//         const int src_h = fb->height;
//         const uint8_t* src = fb->buf;

//         for (int y = 0; y < MODEL_INPUT_SIZE; ++y) {
//             for (int x = 0; x < MODEL_INPUT_SIZE; ++x) {
//                 // 最近邻采样：将 160x120 → 64x64
//                 int src_x = x * src_w / MODEL_INPUT_SIZE;
//                 int src_y = y * src_h / MODEL_INPUT_SIZE;
//                 int index = (src_y * src_w + src_x) * 2; // 每像素2字节(RGB565)

//                 // 提取 RGB565 中的灰度近似
//                 uint8_t byte1 = src[index];
//                 uint8_t byte2 = src[index + 1];

//                 // 解码 RGB565 → 灰度
//                 uint8_t r = (byte2 & 0xF8);
//                 uint8_t g = ((byte2 & 0x07) << 5) | ((byte1 & 0xE0) >> 3);
//                 uint8_t b = (byte1 & 0x1F) << 3;
//                 uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;

//                 // 写入模型输入张量
//                 int input_index = y * MODEL_INPUT_SIZE + x;
//                 input->data[input_index] = gray / 255.0f;  // 归一化为 float32
//             }
//         } 
    
//     esp_camera_fb_return(fb);
     
// #else
//         for (int y = 0; y < MODEL_INPUT_SIZE; ++y) {
//             for (int x = 0; x < MODEL_INPUT_SIZE; ++x) {
                 

//                 // 写入模型输入张量
//                 uint8_t gray =1;
//                 int input_index = y * MODEL_INPUT_SIZE + x;
//                 input->data[input_index] = gray / 255.0f;  // 归一化为 float32
//             }
//         } 
// #endif

//         // 转换图像为 float32 输入 (灰度)
//         //for (int i = 0; i < fb->len && i < input->bytes / sizeof(float); ++i) {
//         //    input->data.f[i] = fb->buf[i] / 255.0f;
//         //}
        
//         interpreter.Invoke();
//         float* feat = output->data;

//         float min_dist = INFINITY;
//         int pred = -1;
//         for (int c = 0; c < NUM_CLASSES; ++c) {
//             float dist = 0.0f;
//             for (int j = 0; j < EMBEDDING_DIM; ++j) {
//                 float diff = feat[j] - class_prototypes[c][j];
//                 dist += diff * diff;
//             }
//             if (dist < min_dist) {
//                 min_dist = dist;
//                 pred = c;
//             }
//         }

//         printf("✅ Result: %s, dist=%.3f\n", class_names[pred], min_dist);

//         char json[128];
//         snprintf(json, sizeof(json),
//                  "{\"class\":\"%s\", \"distance\":%.3f}",
//                  class_names[pred], min_dist);
//         mqtt_send_result(json);

//         vTaskDelay(300000 / portTICK_PERIOD_MS);
//     }
// }
