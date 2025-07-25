idf_component_register(
        SRCS  "wifi_apsta.cpp" "encoder_model.cc" "mqtt_upload.cc" "camera_config.cc"
        #SRCS  "main_04mqtt.cpp" 
        PRIV_INCLUDE_DIRS .
        #REQUIRES mqtt
        PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera tflite-micro) 
