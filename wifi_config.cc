
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
#define WIFI_SSID_STA      "1573"
#define WIFI_PASS_STA      "987654321"

#define WIFI_SSID_AP       "ESP32-AP"
#define WIFI_PASS_AP       "12345678"

#define c_wifi_cfg_ap           0
#define c_wifi_cfg_sta          1
#define c_wifi_cfg              c_wifi_cfg_ap

 
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static const char *TAG = "wifi_config";

static void wifi_event_handler(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (id == WIFI_EVENT_AP_START) ESP_LOGI(TAG, "AP启动成功");
    else if (id == WIFI_EVENT_AP_STACONNECTED) ESP_LOGI(TAG, "设备接入");
    else if (id == WIFI_EVENT_AP_STADISCONNECTED) ESP_LOGI(TAG, "设备断开");
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

strcpy((char*)ap_config.ap.ssid, "ESP32AP");
strcpy((char*)ap_config.ap.password, "12345678");
ap_config.ap.max_connection = 4;
ap_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    // wifi_config_t ap_config = {
    //     .ap = {
    //         .ssid = "ESP32AP",
    //         .password = "12345678",
    //         .max_connection = 4,
    //         .authmode = WIFI_AUTH_WPA2_PSK
    //     }
    // };
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

// Wi-Fi事件处理器
static void wifi_apsta_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                esp_wifi_connect();
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGI(TAG, "Disconnected, retrying...");
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

    // 配置 STA
    // wifi_config_t sta_config = {
    //     .sta = {
    //         .ssid = WIFI_SSID_STA,
    //         .password = WIFI_PASS_STA,
    //         .threshold.authmode = WIFI_AUTH_WPA2_PSK,
    //     },
    // };
wifi_config_t sta_config= {};
 strcpy((char *)sta_config.sta.ssid, WIFI_SSID_STA); 
    strcpy((char *)sta_config.sta.password, WIFI_PASS_STA); 
sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;


    // 配置 AP
    // wifi_config_t ap_config = {
    //     .ap = {
    //         .ssid = WIFI_SSID_AP,
    //         .ssid_len = strlen(WIFI_SSID_AP),
    //         .password = WIFI_PASS_AP,
    //         .max_connection = 4,
    //         .authmode = WIFI_AUTH_WPA_WPA2_PSK
    //     },
    // };
  
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
}
  

#if c_wifi_cfg == c_wifi_cfg_sta
static void wifi_event_handler(void* arg, esp_event_base_t event_base, 
                             int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                ESP_LOGI(TAG, "STA启动完成");
                break;
            case WIFI_EVENT_STA_CONNECTED:
                ESP_LOGI(TAG, "已连接到AP");
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGI(TAG, "连接断开，尝试重连...");
                esp_wifi_connect();  // 自动重连
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "获取到IP:" IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_init(void) {
    // 1. 初始化基础网络组件
    esp_netif_init();
    esp_event_loop_create_default();
    
    // 2. 创建默认STA接口并注册事件回调
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    assert(sta_netif);
    
    // 3. 注册WiFi和IP事件处理
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    // 4. 初始化WiFi驱动
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 5. 配置STA参数
    wifi_config_t sta_config = {
        .sta = {
            .ssid = "HONOR80",
            .password = "1234567890",
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,  // 加密方式
            .pmf_cfg = {
                .capable = true,  // 启用WPA3增强安全
                .required = false
            }
        }
    };

    // 6. 设置模式并启动
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    
    // 可选优化参数
    esp_wifi_set_ps(WIFI_PS_NONE);  // 禁用省电模式，提升稳定性
    esp_wifi_set_max_tx_power(70);
    //esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE);  // 指定信道减少干扰
    
    // 7. 启动WiFi并连接
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_connect());  // 主动触发连接

    ESP_LOGI(TAG, "STA模式已启动,正在连接HONOR80...");
}
 
#endif

