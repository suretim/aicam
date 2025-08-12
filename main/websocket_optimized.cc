// websocket_optimized.c
 

#include "websocket_optimized.h"
#include "esp_log.h"
#include "esp_timer.h"  // 添加这一行解决esp_timer_get_time错误
#include <unistd.h>     // 添加这一行解决close错误
#include <string.h>
static const char *TAG = "WS_OPT";

void init_ws_server_context(ws_server_context_t *ctx) {
    memset(ctx, 0, sizeof(ws_server_context_t));
    ctx->mutex = xSemaphoreCreateMutex();
}

void cleanup_ws_server_context(ws_server_context_t *ctx) {
    if(ctx->mutex) {
        vSemaphoreDelete(ctx->mutex);
    }
}

static void add_ws_client(ws_server_context_t *ctx, int sockfd) {
    if(xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        // 檢查是否已存在
        for(int i = 0; i < MAX_WS_CLIENTS; i++) {
            if(ctx->clients[i].sockfd == sockfd) {
                ctx->clients[i].last_active = esp_timer_get_time() / 1000;
                xSemaphoreGive(ctx->mutex);
                return;
            }
        }
        
        // 尋找空位或最舊的客戶端替換
        int oldest_index = 0;
        uint32_t oldest_time = UINT32_MAX;
        int empty_index = -1;
        
        for(int i = 0; i < MAX_WS_CLIENTS; i++) {
            if(ctx->clients[i].sockfd == 0) {
                empty_index = i;
                break;
            }
            if(ctx->clients[i].last_active < oldest_time) {
                oldest_time = ctx->clients[i].last_active;
                oldest_index = i;
            }
        }
        
        int index = (empty_index != -1) ? empty_index : oldest_index;
        
        // 如果需要替換舊客戶端，先關閉連接
        if(ctx->clients[index].sockfd != 0 && ctx->clients[index].sockfd != sockfd) {
            close(ctx->clients[index].sockfd);
            ESP_LOGI(TAG, "Disconnected old client %d to make room", ctx->clients[index].sockfd);
        }
        
        // 添加新客戶端
        ctx->clients[index].sockfd = sockfd;
        ctx->clients[index].last_active = esp_timer_get_time() / 1000;
        ctx->clients[index].enabled = false;
        
        ESP_LOGI(TAG, "New client connected: %d (slot %d)", sockfd, index);
        xSemaphoreGive(ctx->mutex);
    }
}

static void remove_ws_client(ws_server_context_t *ctx, int sockfd) {
    if(xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        for(int i = 0; i < MAX_WS_CLIENTS; i++) {
            if(ctx->clients[i].sockfd == sockfd) {
                ctx->clients[i].sockfd = 0;
                ESP_LOGI(TAG, "Client %d disconnected", sockfd);
                break;
            }
        }
        xSemaphoreGive(ctx->mutex);
    }
}

static void enable_ws_client(ws_server_context_t *ctx, int sockfd) {
    if(xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        for(int i = 0; i < MAX_WS_CLIENTS; i++) {
            if(ctx->clients[i].sockfd == sockfd) {
                ctx->clients[i].enabled = true;
                break;
            }
        }
        xSemaphoreGive(ctx->mutex);
    }
}

static void check_client_timeouts(ws_server_context_t *ctx) {
    uint32_t now = esp_timer_get_time() / 1000;
    
    if(xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        for(int i = 0; i < MAX_WS_CLIENTS; i++) {
            if(ctx->clients[i].sockfd != 0 && 
               (now - ctx->clients[i].last_active) > CLIENT_TIMEOUT_MS) {
                ESP_LOGI(TAG, "Client %d timed out", ctx->clients[i].sockfd);
                close(ctx->clients[i].sockfd);
                ctx->clients[i].sockfd = 0;
            }
        }
        xSemaphoreGive(ctx->mutex);
    }
}

esp_err_t websocket_handler_optimized(httpd_req_t *req) {
    static ws_server_context_t ws_ctx;
    static bool ws_ctx_initialized = false;
    
    if(!ws_ctx_initialized) {
        init_ws_server_context(&ws_ctx);
        ws_ctx_initialized = true;
    }
    
    if(req->method == HTTP_GET) {
        add_ws_client(&ws_ctx, httpd_req_to_sockfd(req));
        return ESP_OK;
    }
    
    httpd_ws_frame_t ws_pkt;
    memset(&ws_pkt, 0, sizeof(httpd_ws_frame_t));
    
    // 接收幀頭部信息
    esp_err_t ret = httpd_ws_recv_frame(req, &ws_pkt, 0);
    if(ret != ESP_OK) {
        ESP_LOGE(TAG, "httpd_ws_recv_frame failed: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // 更新客戶端活動時間
    enable_ws_client(&ws_ctx, httpd_req_to_sockfd(req));
    
    // 處理不同類型的WebSocket幀
    switch(ws_pkt.type) {
        case HTTPD_WS_TYPE_TEXT:
            // 處理文本消息
            break;
            
        case HTTPD_WS_TYPE_BINARY:
            // 處理二進制消息
            break;
            
        case HTTPD_WS_TYPE_CLOSE:
            remove_ws_client(&ws_ctx, httpd_req_to_sockfd(req));
            break;
            
        default:
            break;
    }
    
    // 定期檢查客戶端超時
    static uint32_t last_timeout_check = 0;
    uint32_t now = esp_timer_get_time() / 1000;
    if(now - last_timeout_check > 5000) { // 每5秒檢查一次
        check_client_timeouts(&ws_ctx);
        last_timeout_check = now;
    }
    
    return ESP_OK;
}