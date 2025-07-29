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
//#include "mqtt_client.h"
#include "mqtt_upload.h"
//#include "class_prototypes.h"

#define WIFI_SSID "1573"
#define WIFI_PASS "987654321"


static const char *TAG = "MAIN";

static EventGroupHandle_t wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
// extern  float  feat_out[EMBEDDING_DIM ];

//const char *class_names[] = {"leaf_disease"};

//static esp_mqtt_client_handle_t mqtt_client = NULL;


// ---------------- MQTT 接收 & 指令处理 ----------------
// static esp_err_t mqtt_event_handler_cb0(esp_mqtt_event_handle_t event) {
//     switch (event->event_id) {
//     case MQTT_EVENT_CONNECTED:
//         ESP_LOGI(TAG, "MQTT connected");
//         esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
//         break;

//     case MQTT_EVENT_DATA: {
//         ESP_LOGI(TAG, "Received message on topic: %.*s", event->topic_len, event->topic);
//         ESP_LOGI(TAG, "Message: %.*s", event->data_len, event->data);

//         // 创建一个空终止的字符串副本
//         char *msg = strndup(event->data, event->data_len);
//         if (msg != NULL) {
//             // 简单检查是否包含指令字段
//             if (strstr(msg, "\"command\"") && strstr(msg, "\"capture\"")) {
//                 ESP_LOGI(TAG, "Parsed command: capture");

//                 // 模拟响应：发布推理结果
//                 publish_feature_vector(feat_out, 64,MQTT_TOPIC_PUB);
//                 // char json[128];
//                 // snprintf(json, sizeof(json),
//                 //         "{\"class\":\"%s\", \"confidence\":%.3f}",
//                 //         class_names[0], 0.5);
//                 // esp_mqtt_client_publish(event->client, MQTT_TOPIC_PUB, json, 0, 1, 0);
//                 ESP_LOGI(TAG, "Sent capture response");
//             } else {
//                 ESP_LOGW(TAG, "Unknown or malformed command.");
//             }

//             free(msg);
//         }
//         break;
//     }


//     default:
//         break;
//     }
//     return ESP_OK;
// }

// static void mqtt_event_handler0(void *handler_args, esp_event_base_t base,
//                                int32_t event_id, void *event_data) {
//     mqtt_event_handler_cb0((esp_mqtt_event_handle_t)event_data);
// }


// ---------------- MQTT 发布推理结果 ----------------
// void mqtt_send_result(esp_mqtt_client_handle_t client, const char *payload) {
//     int msg_id = esp_mqtt_client_publish(client, MQTT_TOPIC_PUB, payload, 0, 1, 0);
//     ESP_LOGI("MQTT", "Published to %s: %s (msg_id=%d)", MQTT_TOPIC_PUB, payload, msg_id);
// }

// ---------------- 启动 MQTT 客户端 ----------------
// void start_mqtt_client0() {
//     // 构造唯一 client_id，例如：leaf_detector_1234
//     static char client_id[64];
//     // snprintf(client_id, sizeof(client_id), MQTT_CLIENT_ID_PREFIX "%d", esp_random() % 10000);
//     snprintf(client_id, sizeof(client_id), "mqttx_fefb0396");
    
//     esp_mqtt_client_config_t mqtt_cfg = {
//         .broker = {
//             .address = {
//                 .uri = MQTT_BROKER_URI
//             }
//         },
//         .credentials = {
//             .username = MQTT_USERNAME,
//             .client_id = client_id,
//             .authentication = {
//                 .password = MQTT_PASSWORD
//             }
//         },
//         .session = {
//             .last_will = { 0 },  // 可选：配置 LWT
//             .disable_clean_session = false,
//             .keepalive = MQTT_KEEPALIVE_SECONDS
//         },
//         .network = {
//             .reconnect_timeout_ms = 3000
//         }
//     };

//     mqtt_client = esp_mqtt_client_init(&mqtt_cfg);

//     #define ESP_EVENT_ANY_ID     ((esp_mqtt_event_id_t) -1)
//     esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
//     ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));

//     // 发布一次初始推理结果
//     // char json[128];
//     // snprintf(json, sizeof(json),
//     //          "{\"class\":\"%s\", \"confidence\":%.3f}",
//     //          class_names[0], 0.5);
//     //mqtt_send_result(mqtt_client, json);
//     publish_feature_vector(feat_out, 64,MQTT_TOPIC_PUB);
// }

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

// ---------------- 主函数 ----------------
extern "C" void app_main(void) {
    // 初始化 NVS
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
    start_mqtt_client();
}
