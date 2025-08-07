// video_stream_optimized.c
#include "video_stream_optimized.h"
#include "esp_timer.h"
#include "img_converters.h"
#include <esp_log.h>

static const char *TAG = "VIDEO_STREAM";

void init_video_stream_context(video_stream_context_t *ctx, httpd_handle_t server) {
    memset(ctx, 0, sizeof(video_stream_context_t));
    ctx->server = server;
    ctx->mutex = xSemaphoreCreateMutex();
    ctx->last_stat_time = esp_timer_get_time() / 1000;
}

void free_video_stream_context(video_stream_context_t *ctx) {
    if(ctx->mutex) {
        vSemaphoreDelete(ctx->mutex);
    }
    for(int i = 0; i < FRAME_BUFFER_COUNT; i++) {
        if(ctx->fb[i]) {
            esp_camera_fb_return(ctx->fb[i]);
        }
    }
    if(ctx->jpeg_buf) {
        free(ctx->jpeg_buf);
    }
}

static void update_frame_statistics(video_stream_context_t *ctx) {
    uint32_t now = esp_timer_get_time() / 1000;
    if(now - ctx->last_stat_time >= 1000) {
        ctx->frame_rate = ctx->frame_count;
        ctx->frame_count = 0;
        ctx->last_stat_time = now;
        //ESP_LOGI(TAG, "Current frame rate: %u fps", ctx->frame_rate);
    }
}

static esp_err_t send_frame_to_clients(httpd_handle_t server, uint8_t *buf, size_t len) {
    httpd_ws_frame_t ws_pkt = {
        .final = true,
        .fragmented = false,
        .type = HTTPD_WS_TYPE_BINARY,
        .payload = buf,
        .len = len
    };
    
    // 實際發送代碼需要根據您的客戶端管理邏輯實現
    // 這裡只是一個示例框架
    return ESP_OK;
}

void video_stream_task_optimized(void *pvParameters) {
    video_stream_context_t ctx;
    init_video_stream_context(&ctx, (httpd_handle_t)pvParameters);
    
    while(1) {
        // 獲取新的幀緩衝
        camera_fb_t *new_fb = esp_camera_fb_get();
        if(!new_fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        
        // 更新統計信息
        ctx.frame_count++;
        update_frame_statistics(&ctx);
        
        // 處理JPEG格式幀
        if(new_fb->format == PIXFORMAT_JPEG) {
            if(xSemaphoreTake(ctx.mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                esp_err_t res = send_frame_to_clients(ctx.server, new_fb->buf, new_fb->len);
                xSemaphoreGive(ctx.mutex);
                
                if(res != ESP_OK) {
                    ESP_LOGE(TAG, "Failed to send frame to clients");
                }
            }
            esp_camera_fb_return(new_fb);
        } 
        // 處理其他格式幀
        else {
            uint8_t *rgb_buf = (uint8_t *)heap_caps_malloc(new_fb->width * new_fb->height * 3, MALLOC_CAP_SPIRAM);
            if(rgb_buf) {
                if(fmt2rgb888(new_fb->buf, new_fb->len, new_fb->format, rgb_buf)) {
                    // 可選: 在圖像上繪製疊加層
                    
                    // 轉換為JPEG
                    uint8_t *jpeg_buf = NULL;
                    size_t jpeg_len = 0;
                    if(fmt2jpg(rgb_buf, new_fb->width * new_fb->height * 3, 
                              new_fb->width, new_fb->height, PIXFORMAT_RGB888, 
                              60, &jpeg_buf, &jpeg_len)) {
                        if(xSemaphoreTake(ctx.mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                            esp_err_t res = send_frame_to_clients(ctx.server, jpeg_buf, jpeg_len);
                            xSemaphoreGive(ctx.mutex);
                            
                            if(res != ESP_OK) {
                                ESP_LOGE(TAG, "Failed to send converted frame");
                            }
                        }
                        free(jpeg_buf);
                    }
                }
                free(rgb_buf);
            }
            esp_camera_fb_return(new_fb);
        }
        
        // 動態調整幀率
        vTaskDelay(pdMS_TO_TICKS(DEFAULT_FRAME_DELAY_MS));
    }
    
    free_video_stream_context(&ctx);
    vTaskDelete(NULL);
}