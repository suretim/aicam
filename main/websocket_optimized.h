// websocket_optimized.h
#pragma once
#include "esp_http_server.h"

#define MAX_WS_CLIENTS 5
#define CLIENT_TIMEOUT_MS 10000

typedef struct {
    int sockfd;
    uint32_t last_active;
    bool enabled;
} ws_client_t;

typedef struct {
    ws_client_t clients[MAX_WS_CLIENTS];
    SemaphoreHandle_t mutex;
} ws_server_context_t;

void init_ws_server_context(ws_server_context_t *ctx);
void cleanup_ws_server_context(ws_server_context_t *ctx);
esp_err_t websocket_handler_optimized(httpd_req_t *req);