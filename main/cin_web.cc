// main_app.c
#include "wifi_optimized.h"
#include "video_stream_optimized.h"
#include "websocket_optimized.h"
#include "esp_camera.h"
#include "esp_http_server.h"
#include "nvs_flash.h"

static const char *TAG = "MAIN_APP";
 #if 1
  #define CONFIG_ESP_WIFI_SSID      "1573"
  #define CONFIG_ESP_WIFI_PASSWORD      "987654321"
#else
  #define CONFIG_ESP_WIFI_SSID      "JD803"
 #define CONFIG_ESP_WIFI_PASSWORD      "18825213948"
#endif

 
void start_esp32ap_webserver() {
    // 初始化NVS
    esp_err_t ret = nvs_flash_init();
    if(ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // 初始化WiFi
    wifi_init_sta_with_retry(CONFIG_ESP_WIFI_SSID, CONFIG_ESP_WIFI_PASSWORD);
    xTaskCreate(wifi_monitor_task, "wifi_monitor", 4096, NULL, 5, NULL);
    
    // 初始化相機
    //init_camera();
    
    // 配置HTTP服務器
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 16;
    config.max_open_sockets = MAX_WS_CLIENTS;
    config.lru_purge_enable = true;
    
    httpd_handle_t server = NULL;
    if(httpd_start(&server, &config) == ESP_OK) {
        // 註冊WebSocket處理程序
        httpd_uri_t ws_uri = {
            .uri = "/ws",
            .method = HTTP_GET,
            .handler = websocket_handler_optimized,
            .user_ctx = NULL,
            .is_websocket = true,
            .handle_ws_control_frames = true
        };
        httpd_register_uri_handler(server, &ws_uri);
        
        // 啟動視頻流任務
        xTaskCreate(video_stream_task_optimized, "video_stream", 4096, server, 5, NULL);
        
        ESP_LOGI(TAG, "HTTP server started");
    } else {
        ESP_LOGE(TAG, "Failed to start HTTP server");
    }
}