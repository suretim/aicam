#pragma once

//#include "esp_event.h"
void wifi_init(void) ;
void wifi_init_apsta(void);
void wait_for_wifi_connection() ;
 void wifi_sta_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);
