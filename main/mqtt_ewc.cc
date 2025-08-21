// mqtt_ewc.cpp
#include "mqtt_ewc.h"
#include "esp_log.h"
#include "mqtt_client.h"

#define TAG "MQTT_EWC"
#define MQTT_BROKER_URI "mqtt://broker.hivemq.com"
#define MQTT_TOPIC "plant/ewc_assets"

std::vector<uint8_t> ewc_buffer;
bool received_flag = false;

// 層資訊（依 Python 端 trainable_variables）
std::vector<std::vector<float>> trainable_layers;
std::vector<std::vector<float>> fisher_layers;
std::vector<std::vector<int>> layer_shapes;

  

bool ewc_assets_received() {
    return !ewc_buffer.empty(); // 可依需改成判斷完整檔案大小
}

void parse_ewc_assets() {
    // 根據 layer_shapes 切分 ewc_buffer
    size_t offset = 0;
    for(auto& shape : layer_shapes) {
        size_t len = 1;
        for(auto s : shape) len *= s;
        std::vector<float> arr(ewc_buffer.begin() + offset, ewc_buffer.begin() + offset + len);
        trainable_layers.push_back(arr);
        offset += len;
    }

    // Fisher matrix 同理（Python 端先約定順序與大小）
    for(auto& shape : layer_shapes) {
        size_t len = 1;
        for(auto s : shape) len *= s;
        std::vector<float> arr(ewc_buffer.begin() + offset, ewc_buffer.begin() + offset + len);
        fisher_layers.push_back(arr);
        offset += len;
    }

    received_flag = true;
    ESP_LOGI(TAG,"Parsed EWC assets: %zu layers", trainable_layers.size());
}
