#pragma once

void  get_mqtt_feature(  float *f_in);
void mqtt_app_start();
void start_mqtt_client(void);
void mqtt_send_result(const char* json);
void classifier_set_params(const float *weights, const float *bias, int input_dim, int output_dim);
