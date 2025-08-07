#include <esp_log.h>
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h>
#include <string.h> 
#include "esp_event.h" 
#include "freertos/FreeRTOS.h"
#include "freertos/task.h" 
#include "freertos/semphr.h"

#include "esp_camera.h"
#include "esp_http_server.h" 
#include <sys/socket.h>
#include <esp_netif.h>
#include "esp_wifi.h"   
#include "esp_timer.h"
#include <pthread.h>

static const char *TAG = "config_stream";

// 定義常量
#define MAX_CLIENTS 5
#define MAX_FRAME_SIZE (50*1024)  // 50KB

// 客戶端優先級定義
typedef enum {
    CLIENT_PRIORITY_HIGH = 0,  // 例如控制端
    CLIENT_PRIORITY_MEDIUM,    // 例如顯示端
    CLIENT_PRIORITY_LOW        // 例如備份端
} client_priority_t;

// 網絡狀態結構
typedef struct {
    float avg_rtt;
    float packet_loss;
    uint32_t last_update;
} network_status_t;

// 客戶端socket結構
typedef struct {
    int sockfd;
    int en;
    unsigned int tmr;
    client_priority_t priority;
} st_client_sockets;

// 幀類型定義
typedef enum {
    FRAME_TYPE_KEY = 0,    // 完整幀
    FRAME_TYPE_DELTA       // 差異幀
} frame_type_t;

// 全局變量
static st_client_sockets client_sockets[MAX_CLIENTS];
static SemaphoreHandle_t web_send_mutex = NULL;
static uint32_t frame_rate = 0;

// 函數聲明
static frame_type_t determine_frame_type(unsigned int frame_cnt);
static void set_client_priority(int sockfd, client_priority_t priority);
static esp_err_t send_batch_frames(httpd_handle_t server, uint8_t *buf, size_t len);
static int count_connected_clients();
static void reduce_video_quality();
static void restore_video_quality();
static bool encode_delta_frame(uint8_t *current, uint8_t *previous, size_t size, uint8_t **out, size_t *out_len);
static void adjust_frame_rate(int connected_clients, float *target_delay);
static void update_network_status(network_status_t *status, float new_rtt, bool packet_lost);
static bool should_reduce_quality(network_status_t *status);
static void update_statistics(unsigned int *frame_tmr, unsigned int *frame_cnt, uint32_t *frame_rate);
void video_stream_task(void *pvParameters) ;
// 初始化函數
void video_stream_init(httpd_handle_t server) {
    web_send_mutex = xSemaphoreCreateMutex();
    if (!web_send_mutex) {
        ESP_LOGE(TAG, "Failed to create mutex");
        return;
    }
    
    // 初始化客戶端socket數組
    memset(client_sockets, 0, sizeof(client_sockets));
    
    // 創建視頻流任務
    xTaskCreate(video_stream_task, "video_stream", 4096, server, 5, NULL);
}

// 確定幀類型
static frame_type_t determine_frame_type(unsigned int frame_cnt) {
    return (frame_cnt % 10 == 0) ? FRAME_TYPE_KEY : FRAME_TYPE_DELTA;
}

// 設置客戶端優先級
static void set_client_priority(int sockfd, client_priority_t priority) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (client_sockets[i].sockfd == sockfd) {
            client_sockets[i].priority = priority;
            break;
        }
    }
}

// 批量發送幀
static esp_err_t send_batch_frames(httpd_handle_t server, uint8_t *buf, size_t len) {
    if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) != pdTRUE) {
        return ESP_FAIL;
    }
    
    esp_err_t ret = ESP_OK;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (client_sockets[i].sockfd <= 0 || client_sockets[i].en == 0) continue;
        
        httpd_ws_frame_t ws_pkt = {
            .final = true,
            .fragmented = false,
            .type = HTTPD_WS_TYPE_BINARY,
            .payload = buf,
            .len = len
        };
        
        if (httpd_ws_send_frame_async(server, client_sockets[i].sockfd, &ws_pkt) != ESP_OK) {
            close(client_sockets[i].sockfd);
            ESP_LOGI(TAG, "Client %d disconnected", client_sockets[i].sockfd);
            client_sockets[i].sockfd = 0;
            ret = ESP_FAIL;
        }
    }
    
    xSemaphoreGive(web_send_mutex);
    return ret;
}

// 計算已連接客戶端數量
static int count_connected_clients() {
    int count = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (client_sockets[i].sockfd > 0 && client_sockets[i].en) {
            count++;
        }
    }
    return count;
}

// 降低視頻質量
static void reduce_video_quality() {
    // 這裡可以實現降低分辨率或幀率的邏輯
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_framesize(s, FRAMESIZE_QVGA);  // 設置為較低分辨率
    }
    ESP_LOGI(TAG, "Reducing video quality due to network conditions");
}

// 恢復視頻質量
static void restore_video_quality() {
    // 恢復到原始質量設置
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_framesize(s, FRAMESIZE_VGA);  // 恢復到較高分辨率
    }
    ESP_LOGI(TAG, "Restoring video quality");
}

// 編碼差異幀
static bool encode_delta_frame(uint8_t *current, uint8_t *previous, size_t size, uint8_t **out, size_t *out_len) {
    *out = (uint8_t *)malloc(size);
    if (!*out) return false;
    
    for (size_t i = 0; i < size; i++) {
        (*out)[i] = current[i] - previous[i];
    }
    
    *out_len = size;
    return true;
}

// 調整幀率
static void adjust_frame_rate(int connected_clients, float *target_delay) {
    static float base_delay = 50.0f;  // 基礎延遲(ms)
    static float max_delay = 200.0f;  // 最大延遲(ms)
    
    float new_delay = base_delay * (1 + connected_clients * 0.2f);
    
    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        if (ap_info.rssi < -80) {  // 信號弱
            new_delay *= 1.5f;
        }
    }
    
    *target_delay = (new_delay > max_delay) ? max_delay : new_delay;
}

// 更新網絡狀態
static void update_network_status(network_status_t *status, float new_rtt, bool packet_lost) {
    float alpha = 0.2f;  // 平滑因子
    
    status->avg_rtt = alpha * new_rtt + (1 - alpha) * status->avg_rtt;
    
    if (packet_lost) {
        status->packet_loss = alpha * 1.0f + (1 - alpha) * status->packet_loss;
    } else {
        status->packet_loss = (1 - alpha) * status->packet_loss;
    }
    
    status->last_update = xTaskGetTickCount();
}

// 判斷是否需要降低質量
static bool should_reduce_quality(network_status_t *status) {
    return (status->avg_rtt > 100.0f) || (status->packet_loss > 0.1f);
}

// 更新統計信息
static void update_statistics(unsigned int *frame_tmr, unsigned int *frame_cnt, uint32_t *frame_rate) {
    static unsigned int last_frame_cnt = 0;
    static unsigned int last_time = 0;
    
    unsigned int now = xTaskGetTickCount();
    if (now - last_time >= 1000) {  // 每秒計算一次幀率
        *frame_rate = (*frame_cnt - last_frame_cnt) * 1000 / (now - last_time);
        last_frame_cnt = *frame_cnt;
        last_time = now;
    }
}

// 視頻流主任務
void video_stream_task(void *pvParameters) {
    static unsigned int frame_tmr = xTaskGetTickCount();
    static unsigned int frame_cnt = 0;
    static uint8_t *prev_frame = NULL;
    static size_t prev_frame_size = 0;
    
    httpd_handle_t server = (httpd_handle_t)pvParameters;
    network_status_t net_status = {0, 0, 0};
    float target_delay = 50.0f;

    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        uint8_t *current_frame = fb->buf;
        size_t frame_size = fb->len;

        int connected = count_connected_clients();
        adjust_frame_rate(connected, &target_delay);
        
        frame_type_t ftype = determine_frame_type(frame_cnt);
        
        uint8_t *delta_frame = NULL;
        size_t delta_len = 0;
        
        if (ftype == FRAME_TYPE_DELTA && prev_frame != NULL) {
            if (encode_delta_frame(current_frame, prev_frame, frame_size, &delta_frame, &delta_len)) {
                send_batch_frames(server, delta_frame, delta_len);
                free(delta_frame);
            } else {
                ftype = FRAME_TYPE_KEY;
            }
        }
        
        if (ftype == FRAME_TYPE_KEY) {
            if (prev_frame_size != frame_size) {
                free(prev_frame);
                prev_frame = (uint8_t*)malloc(frame_size);
                prev_frame_size = frame_size;
            }
            memcpy(prev_frame, current_frame, frame_size);
            
            send_batch_frames(server, current_frame, frame_size);
        }
        
        // 模擬網絡狀態更新
        update_network_status(&net_status, 50.0f + (rand() % 100), (rand() % 100) < 5);
        
        if (should_reduce_quality(&net_status)) {
            reduce_video_quality();
        } else {
            restore_video_quality();
        }
        
        esp_camera_fb_return(fb);
        frame_cnt++;
        
        update_statistics(&frame_tmr, &frame_cnt, &frame_rate);
        vTaskDelay(pdMS_TO_TICKS(target_delay));
    }
    
    if (prev_frame) free(prev_frame);
    vTaskDelete(NULL);
}

//static u_int8_t tensor_state=0;
// u_int8_t get_tensor_state(void)
// {
//    return tensor_state;
// }
#if 0

void video_stream_task(void *pvParameters) {
    static unsigned int frame_tmr = 0;
    static unsigned int frame_cnt = 0;

    httpd_handle_t server = (httpd_handle_t)pvParameters;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    esp_err_t res = ESP_OK;
    
    frame_tmr = tick_get(); 
     
    while (1) {    
        //tensor_state=0;    
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
            //tensor_state=1;
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
                    //tensor_state=1;
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
                //tensor_state=1;
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
#endif
