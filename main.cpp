#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "mqtt_client.h"
#include "class_prototypes.h"

#include "esp_camera.h" 
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
//#include "tflite_model.h"
//#include "tensorflow/lite/c/common.h"
#define WIFI_SSID "1573"
#define WIFI_PASS "987654321"

#define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"
#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_TOPIC_PUB "smartagriculture1/leaf_detection"
#define MQTT_TOPIC_SUB "smartagriculture1/server"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
 
#define MQTT_KEEPALIVE_SECONDS 60

static const char *TAG = "MAIN";

static EventGroupHandle_t wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

#define CAM_PIN_PWDN    -1  //power down is not used
#define CAM_PIN_RESET   -1  //software reset will be performed
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5

#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11

#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13 

extern const unsigned char encoder_model_float[];
extern const int encoder_model_float_len;
 
static esp_mqtt_client_handle_t mqtt_client = NULL;

// ---------------- Wi-Fi 事件回调 ----------------
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Disconnected. Retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// ---------------- MQTT 接收 & 指令处理 ----------------
static esp_err_t mqtt_event_handler_cb(esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
        break;

    case MQTT_EVENT_DATA: {
        ESP_LOGI(TAG, "Received message on topic: %.*s", event->topic_len, event->topic);
        ESP_LOGI(TAG, "Message: %.*s", event->data_len, event->data);

        // 创建一个空终止的字符串副本
        char *msg = strndup(event->data, event->data_len);
        if (msg != NULL) {
            // 简单检查是否包含指令字段
            if (strstr(msg, "\"command\"") && strstr(msg, "\"capture\"")) {
                ESP_LOGI(TAG, "Parsed command: capture");

                // 模拟响应：发布推理结果
                char json[128];
                snprintf(json, sizeof(json),
                        "{\"class\":\"%s\", \"confidence\":%.3f}",
                        class_names[0], 0.5);
                esp_mqtt_client_publish(event->client, MQTT_TOPIC_PUB, json, 0, 1, 0);
                ESP_LOGI(TAG, "Sent capture response");
            } else {
                ESP_LOGW(TAG, "Unknown or malformed command.");
            }

            free(msg);
        }
        break;
    }


    default:
        break;
    }
    return ESP_OK;
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
    mqtt_event_handler_cb((esp_mqtt_event_handle_t)event_data);
}

// ---------------- MQTT 发布推理结果 ----------------
void mqtt_send_result(esp_mqtt_client_handle_t client, const char *payload) {
    int msg_id = esp_mqtt_client_publish(client, MQTT_TOPIC_PUB, payload, 0, 1, 0);
    ESP_LOGI("MQTT", "Published to %s: %s (msg_id=%d)", MQTT_TOPIC_PUB, payload, msg_id);
}

// ---------------- 启动 MQTT 客户端 ----------------
void start_mqtt_client(int pd,float score) {
    // 构造唯一 client_id，例如：leaf_detector_1234
    static char client_id[64];
    // snprintf(client_id, sizeof(client_id), MQTT_CLIENT_ID_PREFIX "%d", esp_random() % 10000);
    snprintf(client_id, sizeof(client_id), "mqttx_fefb0396");
    
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = MQTT_BROKER_URI
            }
        },
        .credentials = {
            .username = MQTT_USERNAME,
            .client_id = client_id,
            .authentication = {
                .password = MQTT_PASSWORD
            }
        },
        .session = {
            .last_will = { 0 },  // 可选：配置 LWT
            .disable_clean_session = false,
            .keepalive = MQTT_KEEPALIVE_SECONDS
        },
        .network = {
            .reconnect_timeout_ms = 3000
        }
    };

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);

    #define ESP_EVENT_ANY_ID     ((esp_mqtt_event_id_t) -1)
    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));

    // 发布一次初始推理结果
    char json[128];
    snprintf(json, sizeof(json),
             "{\"class\":\"%s\", \"confidence\":%.3f}",
             class_names[pd], score);
    mqtt_send_result(mqtt_client, json);
}

// ---------------- 等待 WiFi 连接 ----------------
void wait_for_wifi_connection() {
    ESP_LOGI(TAG, "Waiting for WiFi...");
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdTRUE,
                                           pdMS_TO_TICKS(15000));
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi connected.");
    } else {
        ESP_LOGE(TAG, "WiFi connection timed out.");
    }
}

static camera_config_t camera_config = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    //XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

                   // |--用于图像显示  PIXFORMAT_RGB888不支持
    .pixel_format = PIXFORMAT_RGB565,//PIXFORMAT_JPEG,//PIXFORMAT_RGB565, //YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_QVGA,//FRAMESIZE_QVGA,    //QQVGA-UXGA, For ESP32, do not use sizes above QVGA when not JPEG. The performance of the ESP32-S series has improved a lot, but JPEG mode always gives better frame rates.

    .jpeg_quality = 12, //0-63, for OV series camera sensors, lower number means higher quality
    .fb_count = 2,       //When jpeg mode is used, if fb_count more than one, the driver will work in continuous mode.
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

static esp_err_t init_camera(void)
{
    //initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera Init Failed");
        return err;
    }

    return ESP_OK;
}
 
// ---------------- 主函数 ----------------
extern "C" void app_main (void) {
    
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_event_group = xEventGroupCreate();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {};
    strcpy((char *)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char *)wifi_config.sta.password, WIFI_PASS);

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    wait_for_wifi_connection();
 
#define MODEL_INPUT_SIZE 64

//constexpr int kTensorArenaSize = 350 * 1024;  // 约等于模型请求的 327680
//static uint8_t tensor_arena[kTensorArenaSize];
//#define TENSOR_ARENA_SIZE (40 * 1024)
//static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
    const tflite::Model* model = tflite::GetModel(encoder_model_float);
//const tflite::Model* model = reinterpret_cast<const tflite::Model*>(encoder_model);
 
     
//static tflite::MicroErrorReporter micro_error_reporter;
tflite::MicroMutableOpResolver<6> resolver;
resolver.AddConv2D();
resolver.AddRelu();
resolver.AddMaxPool2D();
resolver.AddAveragePool2D(); 
resolver.AddReshape(); 
 
 
// 3. 初始化解释器（不使用 ErrorReporter）

    constexpr int kTensorArenaSize = 350 * 1024;   
    uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);


    interpreter.AllocateTensors();
    // 确保正确分配输入和输出张量


    
TfLiteTensor* input = interpreter.input(0);
if (input == nullptr) {
    ESP_LOGE(TAG, "Input tensor is null!");
    return;
}

TfLiteTensor* output = interpreter.output(0);
if (output == nullptr) {
    ESP_LOGE(TAG, "Output tensor is null!");
    return;
}
 
#if 1
    init_camera();
     while (1) 
     {
        ESP_LOGI("MAIN", "app_main loop");
 
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera capture failed\n");
            continue;
        }
        // 原图像尺寸（视你配置的 frame_size 决定）
        const int src_w = fb->width;
        const int src_h = fb->height;
        const uint8_t* src = fb->buf; 
        float* input_buffer = (float*)(input->data.f);
        for (int y = 0; y < MODEL_INPUT_SIZE; ++y) {
            for (int x = 0; x < MODEL_INPUT_SIZE; ++x) {
                // 最近邻采样：将 160x120 → 64x64
                int src_x = x * src_w / MODEL_INPUT_SIZE;
                int src_y = y * src_h / MODEL_INPUT_SIZE;
                int index = (src_y * src_w + src_x) * 2; // 每像素2字节(RGB565)

                // 提取 RGB565 中的灰度近似
                uint8_t byte1 = src[index];
                uint8_t byte2 = src[index + 1];

                // 解码 RGB565 → 灰度
                uint8_t r = (byte2 & 0xF8);
                uint8_t g = ((byte2 & 0x07) << 5) | ((byte1 & 0xE0) >> 3);
                uint8_t b = (byte1 & 0x1F) << 3;
                uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;

                // 写入模型输入张量
                int input_index = y * MODEL_INPUT_SIZE + x;
                input_buffer[input_index]  = gray / 255.0f;  // 归一化为 float32
            }
        }  
        esp_camera_fb_return(fb);
        interpreter.Invoke();
        float* feat = (float*)output->data.f ;

        float min_dist = INFINITY;
        int pred = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float dist = 0.0f;
            for (int j = 0; j < EMBEDDING_DIM; ++j) {
                float diff = feat[j] - class_prototypes[c][j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                pred = c;
            }
        }
        start_mqtt_client(pred,min_dist);
        vTaskDelay(300000 / portTICK_PERIOD_MS);
    }
#else
 
    float* input_buffer = (float*)(input->data.f);
    for (int y = 0; y < MODEL_INPUT_SIZE; ++y) {
        for (int x = 0; x < MODEL_INPUT_SIZE; ++x) { 
                

            // 写入模型输入张量
            float gray =1;
            int input_index = y * MODEL_INPUT_SIZE + x;
             
            input_buffer[input_index] =  (float )gray / 255.0f;

            //input->data.f[input_index] = gray / 255.0f;  // 归一化为 float32
        }
    } 
#endif
    
    //printf("✅ Result: %s, dist=%.3f\n", class_names[pred], min_dist);

        //char json[128];
        //snprintf(json, sizeof(json),
        //         "{\"class\":\"%s\", \"distance\":%.3f}",
        //         class_names[pred], min_dist);
        //mqtt_send_result(json);
        
      //  vTaskDelay(300000 / portTICK_PERIOD_MS);
    return;
}



void app_main0(void)
{

    //nvs_flash_init();
    ESP_LOGI("MAIN", "app_main started.");
     
 
    init_camera();
     while (1) 
     {
        ESP_LOGI("MAIN", "app_main loop");
 
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera capture failed\n");
            continue;
        } 
 
        // 原图像尺寸（视你配置的 frame_size 决定）
        const int src_w = fb->width;
        const int src_h = fb->height;
        const uint8_t* src = fb->buf; 
        for (int y = 0; y < MODEL_INPUT_SIZE; ++y) {
            for (int x = 0; x < MODEL_INPUT_SIZE; ++x) {
                // 最近邻采样：将 160x120 → 64x64
                int src_x = x * src_w / MODEL_INPUT_SIZE;
                int src_y = y * src_h / MODEL_INPUT_SIZE;
                int index = (src_y * src_w + src_x) * 2; // 每像素2字节(RGB565)

                // 提取 RGB565 中的灰度近似
                uint8_t byte1 = src[index];
                uint8_t byte2 = src[index + 1];

                // 解码 RGB565 → 灰度
                uint8_t r = (byte2 & 0xF8);
                uint8_t g = ((byte2 & 0x07) << 5) | ((byte1 & 0xE0) >> 3);
                uint8_t b = (byte1 & 0x1F) << 3;
                uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;

                // 写入模型输入张量
                int input_index = y * MODEL_INPUT_SIZE + x;
                //input.data[input_index] = gray / 255.0f;  // 归一化为 float32
            }
        } 
    
    esp_camera_fb_return(fb);
       

       

        vTaskDelay(300000 / portTICK_PERIOD_MS);
    }
}
