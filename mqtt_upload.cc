#include "mqtt_client.h"
#include "esp_log.h"
#include "class_prototypes.h"
 #include <string>
#include <sstream>
#include <stdlib.h>
#include "mqtt_upload.h"
#include "classifier.h"

//#define MQTT_BROKER_URI "mqtt://192.168.133.128:1883"

#define TAG "MQTT"
static esp_mqtt_client_handle_t mqtt_client = NULL;
char client_id[64]=MQTT_CLIENT_ID_PREFIX;
 float  feat_out[EMBEDDING_DIM ]= {
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1
};




// 将 float 数组格式化为 JSON 并上传
void publish_feature_vector(  const char* topic) {
    std::stringstream ss;
    ss << "{\"weights\":[";

    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        ss << feat_out[i];
        if (i != EMBEDDING_DIM - 1) ss << ",";
    }

    ss << "]}";

    std::string payload = ss.str();

    int msg_id = esp_mqtt_client_publish(mqtt_client, topic, payload.c_str(), payload.length(), 1, 0);
    if (msg_id != -1) {
        ESP_LOGI("MQTT", "Published feature vector, msg_id=%d", msg_id);
    } else {
        ESP_LOGE("MQTT", "Failed to publish feature vector");
    }
}


#include "cJSON.h"

void handle_mqtt_message(const char *json_str) {
    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("Failed to parse JSON\n");
        return;
    }

    // weights
    cJSON *weights_array = cJSON_GetObjectItem(root, "mqtrx_weights");
    cJSON *bias_array = cJSON_GetObjectItem(root, "mqtrx_bias");

    if (!weights_array || !bias_array) {
        printf("Missing fields in JSON\n");
        cJSON_Delete(root);
        return;
    }

    int weight_len = cJSON_GetArraySize(weights_array);
    int bias_len = cJSON_GetArraySize(bias_array);

    float *weights = (float *)malloc(sizeof(float) * weight_len);
    float *bias =  (float *)malloc(sizeof(float) * bias_len);

    for (int i = 0; i < weight_len; ++i)
        weights[i] = (float)cJSON_GetArrayItem(weights_array, i)->valuedouble;

    for (int i = 0; i < bias_len; ++i)
        bias[i] = (float)cJSON_GetArrayItem(bias_array, i)->valuedouble;

    // ✅ 传给分类器
    classifier_set_params(weights, bias, weight_len, bias_len);

    cJSON_Delete(root);
    free(weights);
    free(bias);
}

// ---------------- MQTT 接收 & 指令处理 ----------------
static esp_err_t mqtt_event_handler_cb(esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
        break;

    case MQTT_EVENT_DATA: {
        ESP_LOGI(TAG, "Received message on topic: %.*s", event->topic_len, event->topic);
        ESP_LOGI(TAG, "Message: %.*s", event->data_len, event->data);

        // 创建一个空终止的字符串副本
        char *msg = strndup(event->data, event->data_len);
        if (msg != NULL) {
            // 简单检查是否包含指令字段
            //if (strstr(msg, "\"capture\"") && strstr(msg, client_id)) {
            if (strstr(msg, "\"mqtrx_\"") ) {
               handle_mqtt_message( msg ); 
            }
            if (strstr(msg, "\"mqttx_\"") ) {
                ESP_LOGI(TAG, "Parsed command: capture");

                // 模拟响应：发布推理结果
#if 1
                publish_feature_vector( MQTT_TOPIC_PUB);
#else
                char json[128];
                snprintf(json, sizeof(json),
                        "{\"class\":\"%s\", \"confidence\":%.3f}",
                        class_names[0], 0.5);
                esp_mqtt_client_publish(event->client, MQTT_TOPIC_PUB, json, 0, 1, 0);
#endif
                ESP_LOGI(TAG, "Sent capture response");
            } else {
                ESP_LOGW(TAG, "Unknown or malformed command.");
            }

            free(msg);
        }
        break;
    }


    default:
        break;
    }
    return ESP_OK;
}



static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
    mqtt_event_handler_cb((esp_mqtt_event_handle_t)event_data);
}

// ---------------- MQTT 发布推理结果 ----------------
void mqtt_send_result(esp_mqtt_client_handle_t client, const char *payload) {
    int msg_id = esp_mqtt_client_publish(client, MQTT_TOPIC_PUB, payload, 0, 1, 0);
    ESP_LOGI("MQTT", "Published to %s: %s (msg_id=%d)", MQTT_TOPIC_PUB, payload, msg_id);
}

  
// ---------------- 启动 MQTT 客户端 ----------------
void start_mqtt_client (void) { 
    // 构造唯一 client_id，例如：leaf_detector_1234s
    
    snprintf(client_id, sizeof(client_id), MQTT_CLIENT_ID_PREFIX "%ld", random() % 10000);
    //snprintf(client_id, sizeof(client_id), "mqttx_fefb0396");
    
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = MQTT_BROKER_URI
            }
        },
        .credentials = {
            .username = MQTT_USERNAME,
            .client_id = client_id,
            .authentication = {
                .password = MQTT_PASSWORD
            }
        },
        .session = {
            .last_will = { 0 },  // 可选：配置 LWT
            .disable_clean_session = false,
            .keepalive = MQTT_KEEPALIVE_SECONDS      
        },
        .network = {
            .reconnect_timeout_ms = 10000
        }
    };
 
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);

    #define ESP_EVENT_ANY_ID     ((esp_mqtt_event_id_t) -1)
    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));
//#if 0
     
//        publish_feature_vector( MQTT_TOPIC_PUB);
     
//#else
//  float min_dist = INFINITY;
//     int pred = 0;   
//     for (int c = 0; c < NUM_CLASSES; ++c) {
//         float dist = 0.0f;
         
//         for (int j = 0; j < EMBEDDING_DIM; ++j) {
//             float diff = feat[ j ] - class_prototypes[c][j];
//             dist += diff * diff;
//         }
//         if (dist < min_dist) {
//             min_dist = dist;
//             pred = c;
//         }
//     }
     
    // 发布推理结果

//     char json[128];
//     snprintf(json, sizeof(json),
//              "{\"class\":\"%s\", \"confidence\":%.3f}",
//              class_names[0], 0.4);
//     mqtt_send_result(mqtt_client, json);
// #endif 
    //ESP_LOGI(TAG," tensor OK %s\r\n",json);
}
