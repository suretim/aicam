
#include "esp_wifi.h"   
#include "esp_event.h" 
//#include <esp_netif.h>
#include <esp_log.h>
#include <string.h>  
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h> 
 
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#if 0
  #define WIFI_SSID_STA      "1573"
  #define WIFI_PASS_STA      "987654321"
#else
  #define WIFI_SSID_STA      "JD803"
 #define WIFI_PASS_STA      "18825213948"
#endif



#define WIFI_SSID_AP       "ESP32-AP"
#define WIFI_PASS_AP       "12345678"
 
 
static EventGroupHandle_t wifi_event_group;
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static const char *TAG = "wifi_config";

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


static void wifi_event_handler(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (id == WIFI_EVENT_AP_START) ESP_LOGI(TAG, "AP启动成功");
    else if (id == WIFI_EVENT_AP_STACONNECTED) ESP_LOGI(TAG, "设备接入");
    else if (id == WIFI_EVENT_AP_STADISCONNECTED) ESP_LOGI(TAG, "设备断开");
}

// ---------------- Wi-Fi 事件回调 ----------------
  void wifi_sta_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
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
// Wi-Fi事件处理器
static void wifi_apsta_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                esp_wifi_connect();
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGI(TAG, "STA Disconnected, retrying...");
                //vTaskSuspend(tensor_task_handle );
                esp_wifi_connect();
                break;
            case WIFI_EVENT_AP_STACONNECTED:
                ESP_LOGI(TAG, "Device connected to AP");
               
                break;
            case WIFI_EVENT_AP_STADISCONNECTED:
                ESP_LOGI(TAG, "Device disconnected from AP");
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wifi_init(void) 
{
    esp_netif_init();
    esp_event_loop_create_default();

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL);
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_config_t ap_config = {};

    strcpy((char*)ap_config.ap.ssid,WIFI_SSID_AP);
    strcpy((char*)ap_config.ap.password,WIFI_PASS_AP);
    ap_config.ap.max_connection = 4;
    ap_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

   
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap_config);
    
    //esp_wifi_set_ps(WIFI_PS_NONE);  // 禁用省电模式，避免频繁协商
    esp_wifi_set_max_tx_power(80);  // 设置最大发射功率（单位0.25dBm）
    //esp_wifi_set_max_tx_power(70);
    //esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G);   //Wi-Fi 4 (802.11n)
    //esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE);  // 指定信道6减少干扰

    esp_wifi_start();

    esp_netif_ip_info_t ip_info;
    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_AP_DEF"), &ip_info);
    ESP_LOGI(TAG, "got ip:" IPSTR "\n", IP2STR(&ip_info.ip));
}

void wifi_init_apsta(void) {
    s_wifi_event_group = xEventGroupCreate();

    // 初始化NVS（必须）
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 注册事件处理器
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_apsta_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_apsta_event_handler, NULL));
 
    wifi_config_t sta_config= {};
    strcpy((char *)sta_config.sta.ssid, WIFI_SSID_STA); 
        strcpy((char *)sta_config.sta.password, WIFI_PASS_STA); 
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
  
 