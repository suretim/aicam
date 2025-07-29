#pragma once
void mqtt_app_start();
void start_mqtt_client(void);
void publish_feature_vector(const float* vec, int len, const char* topic);
void mqtt_send_result(const char* json);
