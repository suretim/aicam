// wifi_optimized.c
#include "wifi_optimized.h"
 #include "esp_wifi.h"   
#include "esp_event.h" 
//#include <esp_netif.h> 
#include <string.h>  
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h> 
 
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
 static EventGroupHandle_t wifi_event_group;
 static EventGroupHandle_t s_wifi_event_group;
 #define WIFI_CONNECTED_BIT BIT0

static const char *TAG = "WIFI_OPT";
static wifi_state_t s_wifi_state = WIFI_STATE_DISCONNECTED;
static int s_retry_count = 0;

static void wifi_event_handler(void* arg, esp_event_base_t event_base, 
                             int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
        s_wifi_state = WIFI_STATE_CONNECTING;
    } 
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t* event = (wifi_event_sta_disconnected_t*) event_data;
        ESP_LOGI(TAG, "Disconnected from AP (reason: %d)", event->reason);
        
        s_wifi_state = WIFI_STATE_DISCONNECTED;
        
        if (s_retry_count < WIFI_MAX_RETRY) {
            esp_wifi_connect();
            s_retry_count++;
            ESP_LOGI(TAG, "Retry to connect to the AP (%d/%d)", s_retry_count, WIFI_MAX_RETRY);
            s_wifi_state = WIFI_STATE_CONNECTING;
        } else {
            ESP_LOGW(TAG, "Max retries reached, waiting %dms before retry", WIFI_RECONNECT_DELAY_MS);
            vTaskDelay(pdMS_TO_TICKS(WIFI_RECONNECT_DELAY_MS));
            s_retry_count = 0;
            esp_wifi_connect();
            s_wifi_state = WIFI_STATE_CONNECTING;
        }
    } 
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_count = 0;
        s_wifi_state = WIFI_STATE_CONNECTED;
    }
}
#define WIFI_SSID_AP       "ESP32-AP"
#define WIFI_PASS_AP       "12345678"
// #define WIFI_CONNECTED_BIT BIT0
static bool wifi_initialized = false;

void wifi_init_sta_with_retry(const char* ssid, const char* password) {
    s_wifi_event_group = xEventGroupCreate();

    // 初始化NVS（必须）
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());

    if (!wifi_initialized) {
        ESP_ERROR_CHECK(esp_event_loop_create_default());
        // Other wifi init code
        wifi_initialized = true;
    }
 
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 注册事件处理器
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
 
    wifi_config_t sta_config= {};
    strcpy((char *)sta_config.sta.ssid, ssid); 
        strcpy((char *)sta_config.sta.password, password); 
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
 
  
    wifi_config_t ap_config= {};
    strcpy((char *)ap_config.ap.ssid, WIFI_SSID_AP); 
    strcpy((char *)ap_config.ap.password, WIFI_PASS_AP);
    ap_config.ap.ssid_len = strlen(WIFI_SSID_AP);
    ap_config.ap.max_connection = 4;
    ap_config.ap.authmode = WIFI_AUTH_WPA_WPA2_PSK;


    if (strlen(WIFI_PASS_AP) == 0) {
        ap_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));  // AP + STA 模式
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi AP+STA 初始化完成");
    
    // 等待连接路由器成功
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "连接路由器成功！");
    } else {
        ESP_LOGE(TAG, "连接失败！");
    }
    
}
  
 
void wifi_init_sta_with_retry0(const char* ssid, const char* password) {
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    assert(sta_netif);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 註冊事件處理器
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    // 配置 WiFi
    // wifi_config_t wifi_config = {
    //     .sta = {
    //         .ssid = "",
    //         .password = "",
    //         .threshold.authmode = WIFI_AUTH_WPA2_PSK,
    //         .sae_pwe_h2e = WPA3_SAE_PWE_BOTH,
    //     },
    // };

// wifi_config_t wifi_config = {0};
// strncpy((char*)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid));
// strncpy((char*)wifi_config.sta.password, password, sizeof(wifi_config.sta.password));
// wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
// wifi_config.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;

    wifi_config_t sta_config= {};
    strcpy((char *)sta_config.sta.ssid, ssid); 
        strcpy((char *)sta_config.sta.password, password); 
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
 
    // strncpy((char*)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid));
    // strncpy((char*)wifi_config.sta.password, password, sizeof(wifi_config.sta.password));

    // 優化 WiFi 參數
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE)); // 禁用省電模式
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi STA initialization finished");
}

wifi_state_t get_wifi_state() {
    return s_wifi_state;
}
void wifi_deinit() {
    esp_wifi_disconnect();
    esp_wifi_stop();
    esp_wifi_deinit();
    esp_event_loop_delete_default();
}
void wifi_monitor_task(void *pvParameters) {
    while(1) {
        if(s_wifi_state == WIFI_STATE_CONNECTED) {
            wifi_ap_record_t ap_info;
            if(esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
                ESP_LOGI(TAG, "RSSI: %d dBm, Channel: %d", ap_info.rssi, ap_info.primary);
                
                // 信號弱時觸發優化措施
                if(ap_info.rssi < -75) {
                    ESP_LOGW(TAG, "Weak signal detected, optimizing connection...");
                    // 可以添加信道切換或其他優化措施
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10000)); // 每10秒檢查一次
    }
}