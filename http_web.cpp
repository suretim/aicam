/**
 * This example takes a picture every 5s and print its size on serial monitor.
 */

// =============================== SETUP ======================================

// 1. Board setup (Uncomment):
// #define BOARD_WROVER_KIT
// #define BOARD_ESP32CAM_AITHINKER
// #define BOARD_ESP32S3_WROOM
// #define BOARD_ESP32S3_GOOUUU

/**
 * 2. Kconfig setup
 *
 * If you have a Kconfig file, copy the content from
 *  https://github.com/espressif/esp32-camera/blob/master/Kconfig into it.
 * In case you haven't, copy and paste this Kconfig file inside the src directory.
 * This Kconfig file has definitions that allows more control over the camera and
 * how it will be initialized.
 */

/**
 * 3. Enable PSRAM on sdkconfig:
 *
 * CONFIG_ESP32_SPIRAM_SUPPORT=y
 *
 * More info on
 * https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/kconfig.html#config-esp32-spiram-support
 */

// ================================ CODE ======================================

#include <esp_log.h>
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h>
#include <string.h> 
#include "esp_event.h" 
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  
#include "esp_camera.h"
#include "esp_http_server.h" 
#include <sys/socket.h>
#include <esp_netif.h>
#include "esp_wifi.h"   

#include "esp_timer.h"
#include <pthread.h>
// support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif 

static const char *TAG = "http_web";

httpd_handle_t server = NULL;

#ifndef c_ret_ok
#define c_ret_ok        0
#define c_ret_nk        1
#endif

static unsigned int tick_get(void)
{
    return esp_timer_get_time() / 1000;
}
static unsigned int tick_cmp(unsigned int tmr, unsigned int tmo)
{
    return (tick_get() - tmr >= tmo) ? c_ret_ok : c_ret_nk;
}
 

#include "esp_http_server.h"
#include "img_converters.h"
#include "cJSON.h"


SemaphoreHandle_t web_send_mutex;

#define c_web_http          0
#define c_web_socket        1
#define c_web_server        c_web_socket

typedef struct {
        int width;
        int height;
        uint8_t * data;
} fb_data_t;

void fb_gfx_fillRect(fb_data_t *fb, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t color)
{
    int32_t line_step = (fb->width - w) * 3;
    uint8_t *data = fb->data + ((x + (y * fb->width)) * 3);
    uint8_t c0 = color >> 16;
    uint8_t c1 = color >> 8;
    uint8_t c2 = color;
    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            data[0] = c0;
            data[1] = c1;
            data[2] = c2;
            data+=3;
        }
        data += line_step;
    }
}

void fb_gfx_drawFastHLine(fb_data_t *fb, int32_t x, int32_t y, int32_t w, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, w, 1, color);
}

void fb_gfx_drawFastVLine(fb_data_t *fb, int32_t x, int32_t y, int32_t h, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, 1, h, color);
}

#if c_web_server == c_web_http
static esp_err_t index_handler(httpd_req_t *req);
static esp_err_t stream_handler(httpd_req_t *req);
static esp_err_t control_handler(httpd_req_t *req);
static const httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = index_handler,
    .user_ctx  = NULL
};

static const httpd_uri_t control_uri = {
    .uri       = "/control",
    .method    = HTTP_GET,
    .handler   = control_handler,
    .user_ctx  = NULL
};

// 注册URI路由
httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler
};

httpd_handle_t start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if(ESP_OK != httpd_start(&server, &config))
        ESP_LOGE(TAG, "httpd_start failed");
    if(ESP_OK != httpd_register_uri_handler(server, &stream_uri))
        ESP_LOGE(TAG, "httpd_register_uri_handler_stream failed");
    if(ESP_OK != httpd_register_uri_handler(server, &index_uri))    
        ESP_LOGE(TAG, "httpd_register_uri_handler_index failed");    
    if(ESP_OK != httpd_register_uri_handler(server, &control_uri))    
        ESP_LOGE(TAG, "httpd_register_uri_handler_control failed");    
    #if c_wifi_cfg == c_wifi_cfg_ap    
    ESP_LOGI("MAIN", "Server ready at http://192.168.4.1");
    #endif
    web_send_mutex = xSemaphoreCreateMutex();
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

extern const uint8_t _binary_index_http_html_start[] asm("_binary_index_http_html_start");
extern const uint8_t _binary_index_http_html_end[] asm("_binary_index_http_html_end");

// 定义URI处理器
static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, 
        (const char *)_binary_index_http_html_start, 
        _binary_index_http_html_end - _binary_index_http_html_start
    );
}

static esp_err_t parse_get(httpd_req_t *req, char **obuf)
{
    char *buf = NULL;
    size_t buf_len = 0;
    
    buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1)
    {
        buf = (char *)malloc(buf_len);
        if (!buf)
        {
            if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
                httpd_resp_send_500(req);
                xSemaphoreGive(web_send_mutex);
            }
            return ESP_FAIL;
        }
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK)
        {
            *obuf = buf;
            return ESP_OK;
        }
        free(buf);
    }
    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
        httpd_resp_send_404(req);
        xSemaphoreGive(web_send_mutex);
    }
    return ESP_FAIL;
}

static esp_err_t control_handler(httpd_req_t *req) {
    char buf[128];
    ESP_LOGI("MAIN", "control_handler run");
    // 1. 获取URL查询字符串
    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) != ESP_OK) {
        if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
            httpd_resp_send_404(req);
            xSemaphoreGive(web_send_mutex);
        }
        return ESP_FAIL;
    }

    // 2. 解析cmd参数
    char cmd[32];
    if (httpd_query_key_value(buf, "cmd", cmd, sizeof(cmd)) != ESP_OK) {
        if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
            httpd_resp_send_404(req);
            xSemaphoreGive(web_send_mutex);
        }
        return ESP_FAIL;
    }

    // 3. 处理命令并响应
    ESP_LOGI(TAG, "Execute command: %s", cmd);
    char resp[128];
    snprintf(resp, sizeof(resp), "{\"status\":\"OK\",\"cmd\":\"%s\"}", cmd);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
        httpd_resp_send(req, resp, strlen(resp));
        xSemaphoreGive(web_send_mutex);
    }
    return ESP_OK;
}

//流处理程序
static esp_err_t stream_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char *part_buf[128];
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    int64_t frame_tmr;
    unsigned int tmr;

    httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");

    while (true)
    {
        frame_tmr = esp_timer_get_time();
        //获取指向帧缓冲区的指针
        fb = esp_camera_fb_get();
        if (!fb)
        {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
            if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
                httpd_resp_send_500(req);
                xSemaphoreGive(web_send_mutex);
            }
            break;
        }
                
        if(fb->format == PIXFORMAT_JPEG)
        {
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            size_t len = snprintf(part_buf, 128,
                "\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n");

            if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
                res = httpd_resp_send_chunk(req, part_buf, len);
                if(res == ESP_OK) {
                    res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
                }
                xSemaphoreGive(web_send_mutex);
            }
            if (res != ESP_OK)
            {
                xTaskCreate(restart_task, "restart_task", 4096, NULL, 5, NULL);
                return ESP_FAIL;
            }
        }
        else
        {
            res = ESP_OK;
            uint8_t *rgb888_buf = heap_caps_malloc(fb->width * fb->height * 3, MALLOC_CAP_SPIRAM);
            if(rgb888_buf != NULL)
            {
                if(fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf) != true)
                {
                    ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
                    esp_camera_fb_return(fb);
                    free(rgb888_buf);
                    res = ESP_FAIL;
                }
                else
                {
                    fb_data_t fb_data;

                    fb_data.width = fb->width;
                    fb_data.height = fb->height;
                    fb_data.data = rgb888_buf;

                    // rectangle box
                    int x, y, w, h;
                    uint32_t color = 0x0000ff00;                
                    x = 40;
                    y = 40;
                    w = 160;
                    h = 120;
                    fb_gfx_drawFastHLine(&fb_data, x, y, w, color);
                    fb_gfx_drawFastHLine(&fb_data, x, y + h - 1, w, color);
                    fb_gfx_drawFastVLine(&fb_data, x, y, h, color);
                    fb_gfx_drawFastVLine(&fb_data, x + w - 1, y, h, color);
                }            
            }

            if(res == ESP_OK)
            {
                _jpg_buf = NULL;                                                                                              //90          
                bool jpeg_converted = fmt2jpg(rgb888_buf, fb->width * fb->height * 3, fb->width, fb->height, PIXFORMAT_RGB888, 60,  &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                free(rgb888_buf);
                if (!jpeg_converted) {
                    ESP_LOGE(TAG, "JPEG compression failed");
                    if (_jpg_buf) free(_jpg_buf);
                }
                else {
                    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
                    tmr = (esp_timer_get_time() - frame_tmr) / 1000;
                    size_t len = snprintf(part_buf, 128,
                        "\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n");
                    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
                        res = httpd_resp_send_chunk(req, part_buf, len);
                        if(res == ESP_OK) {
                            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
                        }
                        xSemaphoreGive(web_send_mutex);
                    }

                    if (_jpg_buf) free(_jpg_buf);

                    if (res != ESP_OK)
                    {
                        xTaskCreate(restart_task, "restart_task", 4096, NULL, 5, NULL);
                        return ESP_FAIL;
                    }
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    return res;
}

#else 
#define MAX_CLIENTS 3
struct st_client_sockets
{
    int sockfd;
    unsigned int tmr;
    unsigned int en;
};
static struct st_client_sockets client_sockets[MAX_CLIENTS];// = {0};

static float frame_rate = 0;

static void add_client(int sockfd) {
    int i;

    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
        for (i = 0; i < MAX_CLIENTS; i++) {
            if (client_sockets[i].sockfd == sockfd) {
                client_sockets[i].tmr = tick_get();
                client_sockets[i].en = 0;
                xSemaphoreGive(web_send_mutex);
                ESP_LOGI(TAG, "Client %d exist", sockfd);
                return;
            }
        }

        for (i = 0; i < MAX_CLIENTS; i++) {
            if (client_sockets[i].sockfd == 0) {
                client_sockets[i].sockfd = sockfd;
                client_sockets[i].tmr =tick_get();
                client_sockets[i].en = 0;
                xSemaphoreGive(web_send_mutex);
                ESP_LOGI(TAG, "Client %d null add", sockfd);
                return;
            }
        }
        if(i >= MAX_CLIENTS) {            
            close(client_sockets[0].sockfd);
            client_sockets[0] = client_sockets[1];
            client_sockets[1] = client_sockets[2];
            client_sockets[2].sockfd = sockfd;            
            client_sockets[2].tmr = tick_get();
            client_sockets[2].en = 0;
            xSemaphoreGive(web_send_mutex);
            ESP_LOGI(TAG, "Client %d full add", sockfd);
        }
    } 
}

static void remove_client(int sockfd) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (client_sockets[i].sockfd == sockfd) {    
            client_sockets[i].sockfd = 0;                
            ESP_LOGI(TAG, "Client %d disconnected", sockfd);
            break;
        }
    }    
}

static void process_command(httpd_req_t *req, const char *command) {
    char response[128];

    response[0] = 0;
    if(strcmp(command, "help") == 0) {
        sprintf(response, "help fps=?");
    }
    else if(strcmp(command, "fps=?") == 0) {
        sprintf(response, "fps=%.1f", frame_rate);
    }    
    else if(strcmp(command, "ws_send_req") == 0) {
        sprintf(response, "ws_send_req");
    }    
    else if(strcmp(command, "ws_send_en") == 0) {
        add_client(httpd_req_to_sockfd(req));
        return;
    }    

    if(response[0] == 0) {
        snprintf(response, 127, "%s", command);
    }

    httpd_ws_frame_t resp = {
        .final = true,
        .fragmented = false,
        .type = HTTPD_WS_TYPE_TEXT,
        .payload = (uint8_t*)response,
        .len = strlen(response)
    };
    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
        httpd_ws_send_frame(req, &resp);
        xSemaphoreGive(web_send_mutex);
    }
}

static void handle_save_image(httpd_req_t *req, httpd_ws_frame_t *ws_pkt) {
    cJSON *root = cJSON_Parse((char*)ws_pkt->payload);
    if (!root) return;
    
    cJSON *action = cJSON_GetObjectItem(root, "action");
    cJSON *filename = cJSON_GetObjectItem(root, "filename");
    cJSON *data = cJSON_GetObjectItem(root, "data");
    
    if (cJSON_IsString(action) && strcmp(action->valuestring, "save_image") == 0 &&
        cJSON_IsString(filename) && cJSON_IsArray(data)) {
                
        const char *response = "Image save request received";
        httpd_ws_frame_t resp = {
            .final = true,
            .fragmented = false,
            .type = HTTPD_WS_TYPE_TEXT,
            .payload = (uint8_t*)response,
            .len = strlen(response)
        };
        if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
            httpd_ws_send_frame(req, &resp);
            xSemaphoreGive(web_send_mutex);
        }
    }
    
    cJSON_Delete(root);
}

static esp_err_t favicon_handler(httpd_req_t *req) {
        return ESP_OK;
}

const char *ws_resp_type[] = {"CONTINUE", "TEXT", "BINARY", "null","null","null","null","null","CLOSE","PING","PONG"};
static esp_err_t websocket_handler(httpd_req_t *req) {
    if (req->method == HTTP_GET) {
        //process_command(req, "ws_send_req");
        add_client(httpd_req_to_sockfd(req));
        return ESP_OK;
    }

    httpd_ws_frame_t ws_pkt;
    uint8_t *buf = NULL;
    memset(&ws_pkt, 0, sizeof(httpd_ws_frame_t));
    
    esp_err_t ret = httpd_ws_recv_frame(req, &ws_pkt, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to receive frame: %s", esp_err_to_name(ret));
        return ret;
    }

    if(ws_pkt.type < sizeof(ws_resp_type)/sizeof(ws_resp_type[0]))
    {
        ESP_LOGI(TAG, "ws_pkt.type=%s", ws_resp_type[ws_pkt.type]);
    }
    else
    {
        ESP_LOGI(TAG, "ws_pkt.type=ERROR");
    }

    if (ws_pkt.type == HTTPD_WS_TYPE_CLOSE) {
        if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
            int sockfd = httpd_req_to_sockfd(req);
            remove_client(sockfd);
            xSemaphoreGive(web_send_mutex);
            return ESP_OK;
        }
    }    

    if (ws_pkt.type == HTTPD_WS_TYPE_TEXT) {
        buf = (uint8_t *)calloc(1, ws_pkt.len + 1);
        if (!buf) return ESP_ERR_NO_MEM;
        
        ws_pkt.payload = buf;
        ret = httpd_ws_recv_frame(req, &ws_pkt, ws_pkt.len);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "ws_pkt.payload=%s", ws_pkt.payload);
            process_command(req, (char*)ws_pkt.payload);
        }
        free(buf);
    }
    else if (ws_pkt.type == HTTPD_WS_TYPE_BINARY) {
        handle_save_image(req, &ws_pkt);
    }
    
    return ESP_OK;
}

static esp_err_t index_handler(httpd_req_t *req) {
    extern const char index_socket_html_start[] asm("_binary_index_socket_html_start");
    extern const char index_socket_html_end[] asm("_binary_index_socket_html_end");
    
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_socket_html_start, index_socket_html_end - index_socket_html_start);
}

void video_stream_task(void *pvParameters) {
    static unsigned int frame_tmr = 0;
    static unsigned int frame_cnt = 0;

    httpd_handle_t server = (httpd_handle_t)pvParameters;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    esp_err_t res = ESP_OK;
    
    frame_tmr = tick_get();
    while (1) {        
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        if(fb->format == PIXFORMAT_JPEG)
        {
            if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (client_sockets[i].sockfd > 0) {                    
                        if(client_sockets[i].en == 0)
                        {
                            if(tick_cmp(client_sockets[i].tmr, 100) == c_ret_ok)     
                                client_sockets[i].en = 1;
                        }
                        else
                        {
                            httpd_ws_frame_t ws_pkt = {
                                .final = true,
                                .fragmented = false,
                                .type = HTTPD_WS_TYPE_BINARY,
                                .payload = fb->buf,
                                .len = fb->len
                            };                
                            if(httpd_ws_send_frame_async(server, client_sockets[i].sockfd, &ws_pkt) != ESP_OK)
                            {
                                close(client_sockets[i].sockfd);
                                ESP_LOGI(TAG, "Client %d disconnected", client_sockets[i].sockfd);
                                client_sockets[i].sockfd = 0;
                            }
                        }
                    }
                }
                xSemaphoreGive(web_send_mutex);
            }
            
            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(50));   //30
        }
        else
        {
            res = ESP_OK;
            uint8_t *rgb888_buf = (uint8_t *)heap_caps_malloc(fb->width * fb->height * 3, MALLOC_CAP_SPIRAM);
            if(rgb888_buf != NULL)
            {
                if(fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf) != true)
                {
                    ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
                    esp_camera_fb_return(fb);
                    free(rgb888_buf);
                    res = ESP_FAIL;
                }
                else
                {
                    fb_data_t fb_data;

                    fb_data.width = fb->width;
                    fb_data.height = fb->height;
                    fb_data.data = rgb888_buf;

                    int x, y, w, h;
                    uint32_t color = 0x0000ff00;                
                    x = 40;
                    y = 40;
                    w = 40;
                    h = 40;
                    fb_gfx_drawFastHLine(&fb_data, x, y, w, color);
                    fb_gfx_drawFastHLine(&fb_data, x, y + h - 1, w, color);
                    fb_gfx_drawFastVLine(&fb_data, x, y, h, color);
                    fb_gfx_drawFastVLine(&fb_data, x + w - 1, y, h, color);
                }            
            }

            if(res == ESP_OK)
            {
                _jpg_buf = NULL;                                                                                              //90        
                bool jpeg_converted = fmt2jpg(rgb888_buf, fb->width * fb->height * 3, fb->width, fb->height, PIXFORMAT_RGB888, 60,  &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                free(rgb888_buf);
                if (!jpeg_converted) {
                    ESP_LOGE(TAG, "JPEG compression failed");
                    if (_jpg_buf) free(_jpg_buf);
                }
                else {
                    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
                        for (int i = 0; i < MAX_CLIENTS; i++) {
                            if (client_sockets[i].sockfd > 0) {                            
                                if(client_sockets[i].en == 0)
                                {
                                    if(tick_cmp(client_sockets[i].tmr, 100) == c_ret_ok)
                                        client_sockets[i].en = 1;
                                }
                                else
                                {
                                    httpd_ws_frame_t ws_pkt = {
                                        .final = true,
                                        .fragmented = false,
                                        .type = HTTPD_WS_TYPE_BINARY,
                                        .payload = _jpg_buf,
                                        .len = _jpg_buf_len
                                    };                
                                    if(httpd_ws_send_frame_async(server, client_sockets[i].sockfd, &ws_pkt) != ESP_OK)
                                    {
                                        close(client_sockets[i].sockfd);
                                        ESP_LOGI(TAG, "Client %d disconnected", client_sockets[i].sockfd);
                                        client_sockets[i].sockfd = 0;
                                    }
                                }
                            }
                        }
                        xSemaphoreGive(web_send_mutex);
                    }
                    if (_jpg_buf) free(_jpg_buf);            
                    vTaskDelay(pdMS_TO_TICKS(1));
                }
            }
        } 
        frame_cnt++;
        if(tick_cmp(frame_tmr, 10000) == c_ret_ok)
        {
            frame_tmr = tick_get();
            frame_rate = frame_cnt / 10.0f;
            frame_cnt = 0;
            ESP_LOGI(TAG, "frame_rate=%.1f", frame_rate);

            wifi_sta_list_t wifi_sta_list = {0};
            if(esp_wifi_ap_get_sta_list(&wifi_sta_list) == ESP_OK)
            {
                for(int i = 0; i < wifi_sta_list.num; i++)
                {
                    ESP_LOGI(TAG, "num:%d,rssi:%d", i, wifi_sta_list.sta[i].rssi);
                }
            }
        }
    }
}

httpd_handle_t start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 16;
    config.max_resp_headers = 16;
    config.max_open_sockets = MAX_CLIENTS;
    config.server_port = 81;
    config.lru_purge_enable = true;       // 启用LRU清理
    config.recv_wait_timeout = 5;         // 接收超时
    config.send_wait_timeout = 5;         // 发送超时    

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t favicon_uri = {
            .uri = "/favicon.ico",
            .method = HTTP_GET,
            .handler = favicon_handler,  // 空处理
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &favicon_uri);        

        httpd_uri_t ws_uri = {
            .uri = "/ws",
            .method = HTTP_GET,
            .handler = websocket_handler,
            .user_ctx = NULL,
            .is_websocket = true,
            .handle_ws_control_frames = true    // 确保能处理网页刷新和网页关闭
        };
        httpd_register_uri_handler(server, &ws_uri);
        
        httpd_uri_t index_uri = {
            .uri = "/",
            .method = HTTP_GET,
            .handler = index_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &index_uri);

        xTaskCreate(video_stream_task, "video_stream", 4096, server, 5, NULL);
    }
    web_send_mutex = xSemaphoreCreateMutex();
    return server;
}
#endif

#if 0
extern "C" void app_main() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    //wifi_init();
    wifi_init_apsta();
    init_camera();
    start_webserver(); 
    start_mqtt_client();
   
}
#endif
