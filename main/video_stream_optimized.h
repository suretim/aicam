// video_stream_optimized.h
#pragma once
#include "esp_http_server.h"
#include "esp_camera.h"

#define MAX_CLIENTS 5
#define FRAME_BUFFER_COUNT 2
#define DEFAULT_FRAME_DELAY_MS 50

typedef struct {
    httpd_handle_t server;
    camera_fb_t *fb[FRAME_BUFFER_COUNT];
    size_t fb_size[FRAME_BUFFER_COUNT];
    uint8_t *jpeg_buf;
    size_t jpeg_buf_len;
    SemaphoreHandle_t mutex;
    uint32_t frame_rate;
    uint32_t frame_count;
    uint32_t last_stat_time;
} video_stream_context_t;

void video_stream_task_optimized(void *pvParameters);
void init_video_stream_context(video_stream_context_t *ctx, httpd_handle_t server);
void free_video_stream_context(video_stream_context_t *ctx);