#pragma once
#define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"

#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_CLIENT_ID_PREFIX "mqttx_" 
#define MQTT_TOPIC_PUB "model/params"
#define MQTT_TOPIC_SUB "capture/mqttx_"
#define MQTT_KEEPALIVE_SECONDS 60
void  get_mqtt_feature(const float *f_in);
void mqtt_app_start();
void start_mqtt_client(void);
void publish_feature_vector(const float* vec, int len, const char* topic);
void mqtt_send_result(const char* json);
