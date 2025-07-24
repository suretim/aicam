#include <esp_log.h>
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h>
#include <string.h>
 

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
// support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

#include "esp_camera.h"
#include "esp_http_server.h"
//#include "protocol_examples_common.h"
#include <sys/socket.h>
//#include <esp_netif.h>

#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"

#include "esp_http_server.h"
#include "img_converters.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "class_prototypes.h"
#include "mqtt_upload.h"
#include "camera_config.h"
#define MODEL_INPUT_SIZE 64

#define WIFI_SSID_STA      "1573"
#define WIFI_PASS_STA      "987654321"

#define WIFI_SSID_AP       "ESP32-AP"
#define WIFI_PASS_AP       "12345678"

static const char *TAG = "wifi_apsta";
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

extern const unsigned char encoder_model_float[];
extern const int encoder_model_float_len;
static u_int8_t tensor_state=0;
static  camera_fb_t *fb = NULL;
typedef struct {
        int width;
        int height;
        uint8_t * data;
} fb_data_t;

static esp_err_t stream_handler(httpd_req_t *req);
// 注册URI路由
httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler
};
TaskHandle_t tensor_task_handle = NULL;

float min_dist = INFINITY;
        int pred = -1; 
//void tensor_task(void)
float g_buffer[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE];
static void tensor_prework(void)
{
     
    const int src_w = fb->width;
    const int src_h = fb->height;
    const uint8_t* src = fb->buf; 

    
    for (int y = 0; y < MODEL_INPUT_SIZE; ++y) {
        for (int x = 0; x < MODEL_INPUT_SIZE; ++x) {
            // 最近邻采样：将 160x120 → 64x64
            int src_x = x * src_w / MODEL_INPUT_SIZE;
            int src_y = y * src_h / MODEL_INPUT_SIZE;
            int index = (src_y * src_w + src_x) * 2; // 每像素2字节(RGB565)

            // 提取 RGB565 中的灰度近似
            uint8_t byte1 = src[index];
            uint8_t byte2 = src[index + 1];

            // 解码 RGB565 → 灰度
            uint8_t r = (byte2 & 0xF8);
            uint8_t g = ((byte2 & 0x07) << 5) | ((byte1 & 0xE0) >> 3);
            uint8_t b = (byte1 & 0x1F) << 3;
            uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;

            // 写入模型输入张量
            int input_index = y * MODEL_INPUT_SIZE + x;
            g_buffer[input_index]  = gray / 255.0f;  // 归一化为 float32
        }
    }  
    
}


static void tensor_task(void)
{
    
    const tflite::Model* model = tflite::GetModel(encoder_model_float);
    //const tflite::Model* model = reinterpret_cast<const tflite::Model*>(encoder_model);
    
        
    //static tflite::MicroErrorReporter micro_error_reporter;
    tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D(); 
    resolver.AddReshape(); 
    
    constexpr int kTensorArenaSize = 350 * 1024;   
    uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
 
    interpreter.AllocateTensors();
    // 确保正确分配输入和输出张量


        
    TfLiteTensor* input = interpreter.input(0);
    if (input == nullptr) {
        //ESP_LOGE(TAG, "Input tensor is null!");
        //vTaskDelete(NULL);
        return  ;
    }

    TfLiteTensor*  output = interpreter.output(0);
    if (output == nullptr) {
        //ESP_LOGE(TAG, "Output tensor is null!");
        //vTaskDelete(NULL);
        return  ;
    } 
 

    float* input_buffer = (float*)(input->data.f);
    for (int y = 0; y < MODEL_INPUT_SIZE*MODEL_INPUT_SIZE; ++y) {  
            input_buffer[y]  = g_buffer[y];    
    }  
    interpreter.Invoke();
        float* feat = (float*)output->data.f ;

          min_dist = INFINITY;
          pred = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float dist = 0.0f;
            for (int j = 0; j < EMBEDDING_DIM; ++j) {
                float diff = feat[j] - class_prototypes[c][j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                pred = c;
            }
        }
        
        //start_mqtt_client(pred,min_dist);
       
        //vTaskDelete(NULL);
}


// Wi-Fi事件处理器
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                esp_wifi_connect();
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGI(TAG, "Disconnected, retrying...");
                //vTaskSuspend(tensor_task_handle );
                esp_wifi_connect();
                break;
            case WIFI_EVENT_AP_STACONNECTED:
                ESP_LOGI(TAG, "Device connected to AP");
               
                break;
            case WIFI_EVENT_AP_STADISCONNECTED:
                ESP_LOGI(TAG, "Device disconnected from AP");
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void fb_gfx_fillRect(fb_data_t *fb, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t color)
{
    int32_t line_step = (fb->width - w) * 3;
    uint8_t *data = fb->data + ((x + (y * fb->width)) * 3);
    uint8_t c0 = color >> 16;
    uint8_t c1 = color >> 8;
    uint8_t c2 = color;
    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            data[0] = c0;
            data[1] = c1;
            data[2] = c2;
            data+=3;
        }
        data += line_step;
    }
}


httpd_handle_t server = NULL;

httpd_handle_t start_webserver() {
   
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if(ESP_OK != httpd_start(&server, &config))
        ESP_LOGE(TAG, "httpd_start failed");
    if(ESP_OK != httpd_register_uri_handler(server, &stream_uri)){  
            ESP_LOGE(TAG, "httpd_register_uri_handler failed"); 
    }
    
    
    ESP_LOGI("MAIN", "Server ready at http://192.168.4.1/stream");
    //ESP_LOGI("MAIN", "Server ready at http://192.168.0.57/stream");
    
    return server;
}

void stop_webserver() {
    if (server) {
        httpd_stop(server); // 阻塞式停止，确保所有连接关闭
        server = NULL;
        ESP_LOGI(TAG, "HTTP server stopped");
        vTaskDelay(500 / portTICK_PERIOD_MS); // 等待资源释放
    }
}

static void restart_task(void *arg) {
   
    stop_webserver();
    start_webserver();
    vTaskDelete(NULL);
}



void fb_gfx_drawFastHLine(fb_data_t *fb, int32_t x, int32_t y, int32_t w, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, w, 1, color);
}

void fb_gfx_drawFastVLine(fb_data_t *fb, int32_t x, int32_t y, int32_t h, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, 1, h, color);
}

void wifi_init_apsta(void) {
    s_wifi_event_group = xEventGroupCreate();

    // 初始化NVS（必须）
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 注册事件处理器
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    // 配置 STA
    // wifi_config_t sta_config = {
    //     .sta = {
    //         .ssid = WIFI_SSID_STA,
    //         .password = WIFI_PASS_STA,
    //         .threshold.authmode = WIFI_AUTH_WPA2_PSK,
    //     },
    // };
wifi_config_t sta_config= {};
 strcpy((char *)sta_config.sta.ssid, WIFI_SSID_STA); 
    strcpy((char *)sta_config.sta.password, WIFI_PASS_STA); 
sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;


    // 配置 AP
    // wifi_config_t ap_config = {
    //     .ap = {
    //         .ssid = WIFI_SSID_AP,
    //         .ssid_len = strlen(WIFI_SSID_AP),
    //         .password = WIFI_PASS_AP,
    //         .max_connection = 4,
    //         .authmode = WIFI_AUTH_WPA_WPA2_PSK
    //     },
    // };
  
wifi_config_t ap_config= {};
 strcpy((char *)ap_config.ap.ssid, WIFI_SSID_AP); 
    strcpy((char *)ap_config.ap.password, WIFI_PASS_AP);
ap_config.ap.ssid_len = strlen(WIFI_SSID_AP);
ap_config.ap.max_connection = 4;
ap_config.ap.authmode = WIFI_AUTH_WPA_WPA2_PSK;


    if (strlen(WIFI_PASS_AP) == 0) {
        ap_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));  // AP + STA 模式
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi AP+STA 初始化完成");
}
  
//流处理程序
static esp_err_t stream_handler(httpd_req_t *req)
{
    
    
    esp_err_t res = ESP_OK;
    char part_buf[128];
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    size_t out_len, out_width, out_height;
    uint8_t *out_buf;
    bool s;
    httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
    
    while (true)
    {
        //获取指向帧缓冲区的指针
        fb = esp_camera_fb_get();
        if (!fb)
        {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
            httpd_resp_send_500(req);
             return ESP_FAIL; 
        }
                
        res = ESP_OK;
        uint8_t *rgb888_buf =(uint8_t *) heap_caps_malloc(fb->width * fb->height * 3, MALLOC_CAP_SPIRAM);
        if(rgb888_buf != NULL)
        {
            if(fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf) != true)
            {
                ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
                esp_camera_fb_return(fb);
                free(rgb888_buf);
                res = ESP_FAIL;
            }
            else
            { 
         
                if( tensor_state==0){
                    
                   // xTaskCreate(tensor_task, "tensor_task", 8192, NULL, 5, &tensor_task_handle);
                   //   xTaskCreate(tensor_task, "tensor_task", 8192, NULL, 5, NULL);
                    tensor_prework();
                    tensor_state=1;//(tensor_state+1)%100;
                    ESP_LOGI(TAG, "tensor_prework %d",tensor_state);
                    //res = ESP_FAIL;
                    //xTaskCreate(restart_task, "restart_task", 4096, NULL, 5, NULL);
                    
                    //return ESP_FAIL;
                    // if(pred>=0){
                    //     fb_data_t fb_data; 
                    //     fb_data.width = fb->width;
                    //     fb_data.height = fb->height;
                    //     fb_data.data = rgb888_buf;
                    //     int x, y, w, h;
                    //     uint32_t color = 0x0000ff00;   
                    //     x = 40+min_dist;
                    //     y = 40+min_dist;
                    //     w = 40+min_dist;
                    //     h = 40+min_dist;
                    //     fb_gfx_drawFastHLine(&fb_data, x, y, w, color);
                    //     fb_gfx_drawFastHLine(&fb_data, x, y + h - 1, w, color);
                    //     fb_gfx_drawFastVLine(&fb_data, x, y, h, color);
                    //     fb_gfx_drawFastVLine(&fb_data, x + w - 1, y, h, color);
                    // }
                      
                }
                //  
 
            }            
        }

        if(res == ESP_OK)
        {
           
             _jpg_buf = NULL;
            bool jpeg_converted = fmt2jpg(rgb888_buf, fb->width * fb->height * 3, fb->width, fb->height, PIXFORMAT_RGB888, 90,  &_jpg_buf, &_jpg_buf_len);
            esp_camera_fb_return(fb);
            free(rgb888_buf);
            if (!jpeg_converted) {
                ESP_LOGE(TAG, "JPEG compression failed");
                if (_jpg_buf) free(_jpg_buf);
            }
            else {
                
                size_t len = snprintf(part_buf, 64,
                    "\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n");
                res = httpd_resp_send_chunk(req, part_buf, len);
                if(res == ESP_OK) {
                    
                    res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
                }
                if (_jpg_buf) free(_jpg_buf);

                if (res != ESP_OK)
                {
                    //xTaskCreate(restart_task, "restart_task", 4096, NULL, 5, NULL);
                    
                    return ESP_FAIL;
                }
            }
         
            
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    return res;
}

extern "C" void  app_main(void) {
    wifi_init_apsta();

    // 等待连接路由器成功
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "连接路由器成功！");
    } else {
        ESP_LOGE(TAG, "连接失败！");
    }
init_camera();
    //start_webserver(); 
    //
    // xTaskCreate(tensor_task, "tensor_task", 8192, NULL, 5, NULL);
    while(true)
    {
        if(tensor_state ==0)
           xTaskCreate(restart_task, "restart_task", 4096, NULL, 5, NULL);
        if(tensor_state ==1){
            
            ESP_LOGI(TAG, "tensor_task start");
            //xTaskCreate(tensor_task, "tensor_task", 8192, NULL, 5, NULL); 
            tensor_task(); 
            tensor_state=0;
            // vTaskDelete(NULL);
        }
         vTaskDelay(30000 / portTICK_PERIOD_MS); 
    }
    // 你可以添加更多任务：如启动 HTTP 服务器等
}
