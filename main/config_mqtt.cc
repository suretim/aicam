#include "mqtt_client.h"
#include "esp_log.h" 
#include <string>
#include <sstream>
#include <stdlib.h> 
#include "classifier.h"
#include "cJSON.h"
#include "esp_wifi.h"   
#include <mbedtls/sha1.h>
//#define INFINITY 1000 
#define DENSE_IN_FEATURES 64
#define DENSE_OUT_CLASSES 3
//#define NUM_CLASSES 2
//#define EMBEDDING_DIM 64
#define TAG "MQTT"
#if 0
    //#define MQTT_BROKER_URI "mqtt://192.168.133.129:1883"
    #define MQTT_BROKER_URI "mqtt://192.168.68.237:1883"
#else
    //#define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"
    #define MQTT_BROKER_URI "mqtt://127.0.0.1:1883"
#endif
#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_CLIENT_ID_PREFIX "mqttx_" 
#define MQTT_TOPIC_PUB "grpc_sub/weights"
//GRPC_SUBSCRIBE = "grpc_sub/weights"

#define MQTT_TOPIC_SUB "federated_model/parameters"
#define MQTT_KEEPALIVE_SECONDS 120


static esp_mqtt_client_handle_t mqtt_client = NULL; 
float  f_out[DENSE_IN_FEATURES ]= {
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1
};
//char client_id[64]=MQTT_CLIENT_ID_PREFIX;
void  get_mqtt_feature(  float *f_in)
{
    for (int i=0;i<DENSE_IN_FEATURES;i++)
    {
        f_out[i]=f_in[i];
    }
   return  ;
}
uint32_t get_client_id()
{
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
      unsigned char hash[20]; // SHA-1 produces a 20-byte hash
    mbedtls_sha1(mac, 6, hash);
    uint32_t hashNumber = 0;
  for(int i=0; i<4; i++) {
    hashNumber = (hashNumber << 8) | hash[i];
  }
    return hashNumber;
}

// 将 float 数组格式化为 JSON 并上传
void publish_feature_vector(int label,int type ) {
    std::stringstream ss;
    //int label=1;
    //int client_id=1;
   uint32_t  client_id=get_client_id();
    ss << "{\"fea_weights\":[";

    for (int i = 0; i < DENSE_IN_FEATURES; ++i) {
        ss << f_out[i];
        if (i != DENSE_IN_FEATURES - 1) ss << ",";
    }
    ss << "],";
    ss << "\"fea_labels_"<< type << "\":[";     
        ss << label; 

    ss << "],";
    ss << "\"client_id\":";     
        ss << client_id ;

    ss << "}";

    std::string payload = ss.str();

    int msg_id = esp_mqtt_client_publish(mqtt_client, MQTT_TOPIC_PUB, payload.c_str(), payload.length(), 1, 0);
      
    if (msg_id != -1) {
        ESP_LOGI("MQTT", "Published feature vector, msg_id=%d", msg_id);
    } else {
        ESP_LOGE("MQTT", "Failed to publish feature vector");
    }
    return;
}


void publish_fisher_vector(int label ) {
    std::stringstream ss;
    //int label=1;
    //int client_id=1;
   uint32_t  client_id=get_client_id();
    ss << "{\"fisher_theta\":[";

    for (int i = 0; i < DENSE_IN_FEATURES; ++i) {
        ss << f_out[i];
        if (i != DENSE_IN_FEATURES - 1) ss << ",";
    }
    ss << "],";
    ss << "\"fisher_matrix\":[";     
     for (int i = 0; i < DENSE_IN_FEATURES; ++i) {
        ss << f_out[i];
        if (i != DENSE_IN_FEATURES - 1) ss << ",";
    } 

    ss << "],";
    ss << "\"client_id\":";     
        ss << client_id ;

    ss << "}";

    std::string payload = ss.str();

    int msg_id = esp_mqtt_client_publish(mqtt_client, MQTT_TOPIC_PUB, payload.c_str(), payload.length(), 1, 0);
      
    if (msg_id != -1) {
        ESP_LOGI("MQTT", "Published feature vector, msg_id=%d", msg_id);
    } else {
        ESP_LOGE("MQTT", "Failed to publish feature vector");
    }
    return;
}


#if 0
int test_protobuf() {
    // 模拟收到 protobuf 编码的数据（你实际中来自 MQTT）
    // 建议使用 Python 端生成真实数据更方便

    // 这里是手动写的一个 buffer 示例
    uint8_t encoded_data[] = {
        0x08, 0x00,                   // param_type = 0 (WEIGHTS)
        0x15, 0x00, 0x00, 0x20, 0x41, // float 10.0
        0x15, 0x00, 0x00, 0x40, 0x41, // float 12.0
        0x15, 0x00, 0x00, 0x48, 0x41, // float 12.5
        0x18, 0x05                    // client_id = 5
    };

    ParsedModelParams params;
    if(decode_model_params( (uint8_t *)encoded_data, sizeof(encoded_data) , &params))
    //if (decode_model_params(encoded_data, sizeof(encoded_data), &params))
    {
        printf("Decoded ParamType: %d\n", params.param_type);
        printf("Client ID: %d\n",(int) params.client_id);
        printf("Values (%zu):\n", params.value_count);
        for (size_t i = 0; i < params.value_count; ++i) {
            printf("  [%zu] = %f\n", i, params.values[i]);
        }
    } else {
        printf("Failed to decode model parameters.\n");
    }

    return 0;
}


// #define MAX_PARAM_LENGTH 1024  // 可按需要扩大

// /* Enum definitions */
// typedef enum _ParamType {
//     ParamType_UNKNOWN = 0,
//     ParamType_ENCODER_WEIGHT = 1,
//     ParamType_ENCODER_BIAS = 2,
//     ParamType_CLASSIFIER_WEIGHT = 3,
//     ParamType_CLASSIFIER_BIAS = 4,
//     ParamType_FULL_MODEL = 5
// } ParamType;
 
// /* Struct definitions */
// typedef struct _ModelParams {
//     ParamType param_type;
//     float * values;
//     int32_t client_id;
// } ModelParams;
// typedef struct {
//     ParamType param_type;
//     float values[MAX_PARAM_LENGTH];
//     size_t value_count;
//     int32_t client_id;
// } ParsedModelParams;



#define MAX_BIAS_SIZE 3  // 根据你的模型类别数调整

// 全局偏置数组
float classifier_bias[MAX_BIAS_SIZE];
float dense_weights[DENSE_IN_FEATURES][DENSE_OUT_CLASSES];
// 处理 classifier bias 函数
void handle_classifier_bias(const float *values, size_t len) {
    if (len > MAX_BIAS_SIZE) {
        ESP_LOGE(TAG, "Bias length %d exceeds max size %d", len, MAX_BIAS_SIZE);
        return;
    }

    ESP_LOGI(TAG, "Updating classifier bias with length %d", len);

    for (size_t i = 0; i < len; i++) {
        classifier_bias[i] = values[i];
        ESP_LOGD(TAG, "bias[%d] = %f", (int)i, values[i]);
    }
} 

void handle_classifier_weight(const float *values, size_t len, const int32_t *shape, size_t shape_len) {
    if (shape_len != 2) {
        ESP_LOGE(TAG, "Invalid shape");
        return;
    }

    int rows = shape[0]; // 例如 64
    int cols = shape[1]; // 例如 3

    if (rows * cols != len) {
        ESP_LOGE(TAG, "Shape mismatch with values_count");
        return;
    }

    ESP_LOGI(TAG, "Updating classifier weights: %d x %d", rows, cols);
    
    // 在此将 values 保存到你的内存结构中
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float w = values[i * cols + j];
            // 假设你有全局数组 float dense_weights[64][3];
            dense_weights[i][j] = w;
        }
    }
}

 
#endif

#if 1
 #include "model_pb_handler.h"



static esp_err_t mqtt_event_handler_cb (esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            //test_protobuf();
            esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
            //publish_feature_vector(1);
            break;
        case MQTT_EVENT_DATA:{
            
            //printf("MQTT payload received: topic=%.*s\n", event->topic_len, event->topic);
            //printf("MQTT payload received %d : topic=%s\n", event->topic_len, event->topic);
            ParsedModelParams params;
            //ModelParams *msg = (ModelParams*)strndup(event->data, event->data_len);
            if(decode_model_params( (uint8_t *)event->data, event->data_len, &params)) {
  
            // 处理参数类型
            switch (params.param_type) {
                case ParamType_CLASSIFIER_WEIGHT:
                    ESP_LOGI(TAG, "Classifier weight received, values: %d", params.value_count);
                    //handle_classifier_weight(params.values, params.values_count, params.shape, params.shape_count);
                    update_classifier_weights_bias(params.values, params.value_count,0);
                    break;
                case ParamType_CLASSIFIER_BIAS:
                    ESP_LOGI(TAG, "Classifier bias received");
                    //handle_classifier_bias(params.values, params.value_count);
                    update_classifier_weights_bias(params.values, params.value_count,1);

                    break;
                case ParamType_ENCODER_WEIGHT:
                    ESP_LOGI(TAG, "Fisher weight received");
                    update_fishermatrix_theta(params.values, params.value_count,0);
                    break;
                case ParamType_ENCODER_BIAS:
                    ESP_LOGI(TAG, "Fisher bias received");
                    update_fishermatrix_theta(params.values, params.value_count,1);
                    break;
                default:
                    ESP_LOGW(TAG, "Unknown or unsupported param_type: %d", params.param_type);
                    break;
            }




            }
            else{
                //printf("MQTT decode error data_len=%d : data=%s\n", event->data_len, event->data);
                printf("MQTT decode error data_len=%d \r\n", event->data_len );
            }
            break;
        }
        default:
            break;
    }
    return ESP_OK;
}
#else



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

     
    classifier_set_params(weights, bias, weight_len, bias_len);

    cJSON_Delete(root);
    free(weights);
    free(bias);
    return;
}


// ---------------- MQTT 接收 & 指令处理 ----------------
static esp_err_t mqtt_event_handler_cb(esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        esp_mqtt_client_subscribe(event->client, MQTT_TOPIC_SUB, 1);
        publish_feature_vector();
        break;

    case MQTT_EVENT_DATA: {
        ESP_LOGI(TAG, "Received message on topic: %.*s", event->topic_len, event->topic);
        ESP_LOGI(TAG, "Message: %.*s", event->data_len, event->data);

        // 创建一个空终止的字符串副本
        char *msg = strndup(event->data, event->data_len);
        if (msg != NULL) {
            // 简单检查是否包含指令字段
            //if (strstr(msg, "\"capture\"") && strstr(msg, client_id)) {
            if (strstr(msg, "\"weights\"") ) {
               handle_mqtt_message( msg ); 
            }
            if (strstr(msg, "\"mqttx_\"") ) {
                ESP_LOGI(TAG, "Parsed command: capture");

                // 模拟响应：发布推理结果
//#if 1
                publish_feature_vector();
// #else
//                 char json[128];
//                 snprintf(json, sizeof(json),
//                         "{\"class\":\"%s\", \"confidence\":%.3f}",
//                         class_names[0], 0.5);
//                 esp_mqtt_client_publish(event->client, MQTT_TOPIC_PUB, json, 0, 1, 0);
// #endif
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
#endif


static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
    mqtt_event_handler_cb((esp_mqtt_event_handle_t)event_data);
    return;
}

// ---------------- MQTT 发布推理结果 ----------------
// void mqtt_send_result(esp_mqtt_client_handle_t client, const char *payload) {
//     int msg_id = esp_mqtt_client_publish(client, MQTT_TOPIC_PUB, payload, 0, 1, 0);
//     ESP_LOGI("MQTT", "Published to %s: %s (msg_id=%d)", MQTT_TOPIC_PUB, payload, msg_id);
// }

  
// ---------------- 启动 MQTT 客户端 ----------------
void start_mqtt_client (void) { 
    // 构造唯一 client_id，例如：leaf_detector_1234s
       char client_id[64];
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
    return;
}