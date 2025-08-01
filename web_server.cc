
#include "esp_http_server.h"
#include "img_converters.h"


httpd_handle_t start_1webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if(ESP_OK != httpd_start(&server, &config))
        ESP_LOGE(TAG, "httpd_start failed");
    if(ESP_OK != httpd_register_uri_handler(server, &stream_uri))
        ESP_LOGE(TAG, "httpd_register_uri_handler failed");
    //ESP_LOGI("MAIN", "Server ready at http://192.168.4.1/stream");
    ESP_LOGI("MAIN", "Server ready at http://192.168.0.57/stream");
    
    return server;
}

void stop_webserver() {
    if (server) {
        httpd_stop(server); // 阻塞式停止，确保所有连接关闭
        server = NULL;
        ESP_LOGI(TAG, "HTTP server stopped");
        vTaskDelay(500 / portTICK_PERIOD_MS); // 等待资源释放
    }
}

static void restart_task(void *arg) {
    stop_webserver();
    start_webserver();
    vTaskDelete(NULL);
}
