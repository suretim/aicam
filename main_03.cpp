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

static const char *TAG = "MAIN";

#define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"
#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_TOPIC    "smartagriculture1/leaf_detection"
#define MQTT_KEEPALIVE_SECONDS 60
#define MQTT_CLIENT_ID_PREFIX "leaf_detector_"


// WiFi 连接成功的标志位
#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t wifi_event_group;

// MQTT class_names 示例
const char *class_names[] = {"leaf_disease"};

// ---------- WiFi 事件处理器 ----------
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

// ---------- 等待 WiFi 连接 ----------
void wait_for_wifi_connection() {
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdTRUE,
                                           pdMS_TO_TICKS(15000));  // 最多等待 15 秒
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi connected successfully.");
    } else {
        ESP_LOGE(TAG, "WiFi connection timed out.");
    }
}

// ---------- MQTT 发布函数 ----------
void mqtt_send_result(esp_mqtt_client_handle_t client, const char *payload) {
    int msg_id = esp_mqtt_client_publish(client, MQTT_TOPIC, payload, 0, 1, 0);
    ESP_LOGI("MQTT", "Published message ID %d to topic %s: %s", msg_id, MQTT_TOPIC, payload);
}

// ---------- 启动 MQTT 客户端 ----------
void start_mqtt_client() {
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

    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    ESP_ERROR_CHECK(esp_mqtt_client_start(client));

    // 构造并发送示例推理结果
    char json_payload[128];
    snprintf(json_payload, sizeof(json_payload),
             "{\"class\":\"%s\", \"confidence\":%.3f}",
             class_names[0], 0.5);  // 可根据实际推理输出替换

    mqtt_send_result(client, json_payload);
}


// ---------- 应用主函数 ----------
extern "C" void app_main(void) {
    // 初始化 NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    // 初始化 TCP/IP、事件循环
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_event_group = xEventGroupCreate();

    // 注册事件处理器
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));

    // 初始化默认 WiFi STA
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 配置 WiFi
    wifi_config_t wifi_config = {};
    strcpy((char *)wifi_config.sta.ssid, "1573");
    strcpy((char *)wifi_config.sta.password, "987654321");

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // 等待连接成功
    wait_for_wifi_connection();

    // 启动 MQTT 客户端
    start_mqtt_client();
}
