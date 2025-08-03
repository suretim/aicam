
if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_GREATER  "5.0") 
#if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_LESS  "5.0") 

idf_component_register(
        PRIV_INCLUDE_DIRS .        
        #for http_web
        SRCS  "main_grpc_client.cpp" "http_web.cpp" "img_inference.cc" "classifier.cc" "camera_config.cc" "wifi_config.cc" "mqtt_upload.cc" "model_data.cc"
        EMBED_FILES "web_content/index_http.html" "web_content/index_socket.html" "web_content/index_ecm.html"
        #for main.cpp
        #SRCS  "wifi_apsta.cpp" "encoder_model.cc" "mqtt_upload.cc" "camera_config.cc"
        #SRCS  "main_mqtt.cpp" "mqtt_upload.cc" 
        #SRCS  "main_04mqtt.cpp"         
       
        #PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera esp-tflite-micro esp_timer cjson lwip esp_eth driver
        PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera tflite-micro json
         
        ) 
else()

idf_component_register(
        PRIV_INCLUDE_DIRS .        
           
        SRCS  "main_wifi_apsta.cpp" "model_data.cc" "mqtt_upload.cc" "camera_config.cc" "classifier.cc" "wifi_config.cc" 
        #SRCS  "main_mqtt.cpp" "mqtt_upload.cc" 
        #SRCS  "main_04mqtt.cpp"         
       
        #PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera esp-tflite-micro esp_timer cjson lwip esp_eth driver
        PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera tflite-micro json
         
        ) 
endif()
#$env:IDF_COMPONENT_MANAGER="enabled"
#https://components.espressif.com/api/