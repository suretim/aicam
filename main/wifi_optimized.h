// wifi_optimized.h
#pragma once
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"

#define WIFI_MAX_RETRY 5
#define WIFI_RECONNECT_DELAY_MS 3000

typedef enum {
    WIFI_STATE_DISCONNECTED = 0,
    WIFI_STATE_CONNECTING,
    WIFI_STATE_CONNECTED
} wifi_state_t;

void wifi_init_sta_with_retry(const char* ssid, const char* password);
wifi_state_t get_wifi_state();
void wifi_monitor_task(void *pvParameters);