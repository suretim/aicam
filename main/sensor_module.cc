
float read_temperature_sensor(void)
{
    return 28.0;
}

float read_humidity_sensor(void)
{
    return 28.0;
}

float read_light_sensor(void)
{
    return 28.0;
}
uint8_t img_0[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE*3];
 

// 传感器数据采集函数
sensor_data_t *get_sensor_data(int8_t *img) {
    static sensor_data_t data;
    // 获取湿度数据
    data.temperature = read_temperature_sensor();
    data.humidity = read_humidity_sensor();
    // 获取光感器数据
    data.light_lux = read_light_sensor();
    // 获取图像数据
    data.camera_frame=img
    //capture_image( data.camera_frame); // 从相机模块获取 64x64 的图像数据
    return &data;
} 
// MQTT 上传推理结果和模型更新
void RespondToDetection(float prediction, float uncertainty) {
    
    //esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    //esp_mqtt_client_start(client);
    
    char payload[128];
    snprintf(payload, sizeof(payload), "{\"prediction\": %f, \"uncertainty\": %f}", prediction, uncertainty);
    
    //esp_mqtt_client_publish(client, MQTT_TOPIC, payload, 0, 1, 0);
    ESP_LOGI(TAG, "Result uploaded: %s", payload);
}
  
 

