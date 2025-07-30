idf_component_register(
        PRIV_INCLUDE_DIRS .
        #for http_web
        SRCS "http_web.cpp"  "camera_config.cc" "wifi_config.cc" "mqtt_upload.cc"
        #PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera esp-tflite-micro esp_timer cjson lwip esp_eth driver
        EMBED_FILES "web_content/index_http.html" "web_content/index_socket.html" "web_content/index_ecm.html"
        #for main.cpp
        #SRCS  "wifi_apsta.cpp" "encoder_model.cc" "mqtt_upload.cc" "camera_config.cc"
        #SRCS  "main_mqtt.cpp" "mqtt_upload.cc" 
        #SRCS  "main_04mqtt.cpp"         
        PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera tflite-micro json
         
        ) 
