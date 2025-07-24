#include "mqtt_client.h"
#include "esp_log.h"
#include "class_prototypes.h"

#define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"
#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_TOPIC_PUB "smartagriculture1/leaf_detection"
#define MQTT_TOPIC_SUB "smartagriculture1/server"
#define MQTT_KEEPALIVE_SECONDS 60

#define TAG "MQTT"
static esp_mqtt_client_handle_t mqtt_client = NULL;

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
