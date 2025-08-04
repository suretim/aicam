
if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_GREATER  "5.0") 
#if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_LESS  "5.0") 

idf_component_register(
        PRIV_INCLUDE_DIRS .        
        #for http_web
        #SRCS  "main_grpc_client.cpp" "http_web.cpp" "img_inference.cc" "camera_config.cc" "config_wifi.cc" "config_mqtt.cc" "model_data.cc" "classifier.cc" 
        SRCS  "main_grpc_client.cpp" "http_web.cpp" "img_inference.cc" "camera_config.cc" "config_wifi.cc" "config_mqtt.cc" "classifier.cc" 
        EMBED_FILES "web_content/index_http.html" "web_content/index_socket.html" "web_content/index_ecm.html" "encoder_model.tflite"
          
        #PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera esp-tflite-micro esp_timer cjson lwip esp_eth driver
        PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera tflite-micro json
        
        ) 
else()

idf_component_register(
        PRIV_INCLUDE_DIRS .        
           
        #SRCS  "main_wifi_apsta.cpp" "model_data.cc" "mqtt_upload.cc" "camera_config.cc" "classifier.cc" "wifi_config.cc" 
        SRCS  "main_mqtt.cpp" "config_mqtt.cc" "config_wifi.cc" 
       
        #PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera esp-tflite-micro esp_timer cjson lwip esp_eth driver
        PRIV_REQUIRES mqtt nvs_flash esp_psram esp_http_server console esp_netif esp_http_server esp_wifi esp32-camera tflite-micro esp-nn json
        EMBED_FILES "web_content/index_http.html" "web_content/index_socket.html" "web_content/index_ecm.html" "encoder_model.tflite"
         
        ) 
endif()
#$env:IDF_COMPONENT_MANAGER="enabled"
#https://components.espressif.com/api/