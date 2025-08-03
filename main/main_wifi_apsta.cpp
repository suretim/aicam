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

#include "wifi_config.h"
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
//#include "class_prototypes.h"
#include "mqtt_upload.h"
#include "camera_config.h"
#include <esp_task_wdt.h>
#include "model_data.h"
#define MODEL_INPUT_SIZE 64


static const char *TAG = "wifi_apsta";
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

  
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
float  feat_out[64 ]; 
float g_buffer[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE];
static void tensor_prework(void)
{
     if (fb == NULL) {
        ESP_LOGE(TAG, "Frame buffer pointer is NULL!");
        return; // 直接返回，避免崩溃
    }

    const int src_w = fb->width;
    const int src_h = fb->height;
    const uint8_t* src = fb->buf;

    if (src == NULL) {
        ESP_LOGE(TAG, "Frame buffer data is NULL!");
        return;
    }
     
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
            g_buffer[input_index]  = gray ;  // 归一化为 float32
        }
    }  
    
}


static void tensor_task(void *arg)
{
    esp_task_wdt_add(NULL);  // 注册到 watchdog
  //vTaskDelay(50 / portTICK_PERIOD_MS);  // 延迟50ms
    const tflite::Model* model = tflite::GetModel(encoder_model_float);
    //const tflite::Model* model = reinterpret_cast<const tflite::Model*>(encoder_model);
    
        
    //static tflite::MicroErrorReporter micro_error_reporter;
    tflite::MicroMutableOpResolver<6> resolver;
   resolver.AddConv2D();
resolver.AddRelu();
resolver.AddMaxPool2D();
resolver.AddAveragePool2D();
resolver.AddReshape();
resolver.AddFullyConnected(); 
    
      int kTensorArenaSize = encoder_model_float_len + (64 * 1024) ;  //800 * 1024;   
    uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM); 
 
// Check if the allocation was successful
if (tensor_arena == NULL) {
    ESP_LOGE(TAG, "Failed to allocate memory for tensor arena in PSRAM!");
    // Handle failure (e.g., attempt to reduce tensor arena size or free other memory)
} else {
    ESP_LOGI(TAG, "Tensor arena allocated in PSRAM.");
}
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
 
    interpreter.AllocateTensors();
    // 确保正确分配输入和输出张量


        
    TfLiteTensor* input = interpreter.input(0);
    if (input == nullptr) {
        ESP_LOGE(TAG, "Input tensor is null!");
        vTaskDelete(NULL);
        return  ;
    }

    TfLiteTensor*  output = interpreter.output(0);
    if (output == nullptr) {
        ESP_LOGE(TAG, "Output tensor is null!");
        vTaskDelete(NULL);
        return  ;
    } 
 

    float* input_buffer = (float*)(input->data.f);
    for (int y = 0; y < MODEL_INPUT_SIZE*MODEL_INPUT_SIZE; ++y) {  
            input_buffer[y]  = g_buffer[y];    
    }  
    esp_task_wdt_reset(); 
    interpreter.Invoke();
    esp_task_wdt_reset(); 
    ESP_LOGI(TAG, "Inference Done tensor_state=%d",tensor_state);    
      
    // = (float*)output->data.f ;
    for (int j = 0; j < output->dims->size; ++j) {
        printf("Dimension %d: %d\r\n", j, output->dims->data[j]);
    }
    int num_features = 1;
    for (int i = 0; i < output->dims->size; ++i) {
        num_features *= output->dims->data[i];
    }
    for (int j = 0; j < num_features; ++j) {
        feat_out[j] = output->data.f[j] ;
    }
 
     //tensor_state =5;  // 处理完成，通知主任务可继续
    
    // ESP_LOGI(TAG, "Class Done tensor_state=%d",tensor_state);    
     
    heap_caps_free(tensor_arena);
    esp_task_wdt_delete(NULL);  // 可选：退出前注销 watchdog
    vTaskDelete(NULL);  // 删除自己
    
}


// Wi-Fi事件处理器
// static void wifi_event_handler(void *arg, esp_event_base_t event_base,
//                                int32_t event_id, void *event_data) {
//     if (event_base == WIFI_EVENT) {
//         switch (event_id) {
//             case WIFI_EVENT_STA_START:
//                 esp_wifi_connect();
//                 break;
//             case WIFI_EVENT_STA_DISCONNECTED:
//                 ESP_LOGI(TAG, "Disconnected, retrying...");
//                 //vTaskSuspend(tensor_task_handle );
//                 esp_wifi_connect();
//                 break;
//             case WIFI_EVENT_AP_STACONNECTED:
//                 ESP_LOGI(TAG, "Device connected to AP");
               
//                 break;
//             case WIFI_EVENT_AP_STADISCONNECTED:
//                 ESP_LOGI(TAG, "Device disconnected from AP");
//                 break;
//         }
//     } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
//         ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
//         ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
//         xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
//     }
// }

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

static void start_webserver(void *arg ) {
    if (server) {
        httpd_stop(server); // 阻塞式停止，确保所有连接关闭
        server = NULL;
        ESP_LOGI(TAG, "HTTP server stopped");
        vTaskDelay(500 / portTICK_PERIOD_MS); // 等待资源释放
    }
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if(ESP_OK != httpd_start(&server, &config))
        ESP_LOGE(TAG, "httpd_start failed");
    
    
    if(ESP_OK != httpd_register_uri_handler(server, &stream_uri)){  
            ESP_LOGE(TAG, "httpd_register_uri_handler failed"); 
    }  
    ESP_LOGI("MAIN", "Server ready at http://192.168.4.1/stream");
    //ESP_LOGI("MAIN", "Server ready at http://192.168.0.57/stream");
    
    vTaskDelete(NULL);
    return  ;
}
  
 


void fb_gfx_drawFastHLine(fb_data_t *fb, int32_t x, int32_t y, int32_t w, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, w, 1, color);
}

void fb_gfx_drawFastVLine(fb_data_t *fb, int32_t x, int32_t y, int32_t h, uint32_t color)
{
    fb_gfx_fillRect(fb, x, y, 1, h, color);
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
        
        //fb = esp_camera_fb_get();
        if (!fb)
        {
            ESP_LOGE(TAG, "Camera capture failed");
            //res = ESP_FAIL;
            httpd_resp_send_500(req);
            vTaskDelay(pdMS_TO_TICKS(10));
             return ESP_FAIL; 
        }
        // if(tensor_state!=1){
        //     vTaskDelay(pdMS_TO_TICKS(100));
        //    continue;     
        // }
        //res = ESP_OK;
        uint8_t *rgb888_buf =(uint8_t *) heap_caps_malloc(fb->width * fb->height * 3, MALLOC_CAP_SPIRAM);
        if(rgb888_buf == NULL){
            return ESP_FAIL; 
        }
        if(fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf) != true)
        {
            ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
            //esp_camera_fb_return(fb);
            free(rgb888_buf);
            vTaskDelay(pdMS_TO_TICKS(10));
            //res = ESP_FAIL;
            return ESP_FAIL;
        }
            //else
            //{ 
         
                 
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
                      
                
 
        //    }            
        //}

        //if(res == ESP_OK)
        //{
           
                _jpg_buf = NULL;
                bool jpeg_converted = fmt2jpg(rgb888_buf, fb->width * fb->height * 3, fb->width, fb->height, PIXFORMAT_RGB888, 90,  &_jpg_buf, &_jpg_buf_len);
                //esp_camera_fb_return(fb);
                free(rgb888_buf);
                if (!jpeg_converted) {
                    ESP_LOGE(TAG, "JPEG compression failed");
                    if (_jpg_buf) 
                        free(_jpg_buf);
                    vTaskDelay(pdMS_TO_TICKS(10));
                    return ESP_FAIL;
                }
                //else {
                    
                    size_t len = snprintf(part_buf, 64,
                        "\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n");
                    res = httpd_resp_send_chunk(req, part_buf, len);
                    if(res == ESP_OK) {
                        
                        res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
                    }
                    if (_jpg_buf) free(_jpg_buf);

                    //if (res != ESP_OK)
                    //{
                        //xTaskCreate(start_webserver, "start_webserver", 4096, NULL, 5, NULL);
                    //    vTaskDelay(pdMS_TO_TICKS(10));
                    //    return ESP_FAIL;
                    //}
                //}
                //if( tensor_state==0){  
                    //tensor_prework();
                //    tensor_state=1;//(tensor_state+1)%100;
                    //ESP_LOGI(TAG, "tensor_prework %d",tensor_state); 
                //}
            //}
            //tensor_state=2;
            vTaskDelay(pdMS_TO_TICKS(10));
         
    }
    return res;
}
 
void  main_task(void *arg) {  
    esp_task_wdt_add(NULL);
    tensor_state=0;
    while(true)
    {  
        if( tensor_state==0){
            esp_task_wdt_reset(); 
             fb = esp_camera_fb_get();
            if (!fb)
            {
                ESP_LOGE(TAG, "Camera capture failed");
                
                //httpd_resp_send_500(req);
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
                // return ESP_FAIL; 
            }
            tensor_state=2;
        }
        if( tensor_state==2){   
            esp_task_wdt_reset();      
             tensor_prework();
            tensor_state=3; 
            ESP_LOGI(TAG, "tensor_prework %d",tensor_state);
            esp_camera_fb_return(fb);
        }
        if(tensor_state ==3){
            tensor_state = 4;  // 表示"正在运行"，避免重复启动
             ESP_LOGI(TAG, "tensor_task start %d",tensor_state);
             esp_task_wdt_reset(); 
            xTaskCreate(tensor_task, "tensor_task", 8192, NULL,6, NULL); //  异步运行
            vTaskDelay(100 / portTICK_PERIOD_MS);  // 延时100ms，避免阻塞

            esp_task_wdt_reset(); 
                
        } 
        esp_task_wdt_reset(); 
        vTaskDelay(pdMS_TO_TICKS(10));
        // vTaskDelay(30000 / portTICK_PERIOD_MS); 
    } 
    vTaskDelete(NULL);
} 
void init_task_watchdog_once()
{
    const esp_task_wdt_config_t twdt_config = {
        .timeout_ms = 5000,
        .idle_core_mask = (1 << 0),
        .trigger_panic = true
    };

    esp_err_t err = esp_task_wdt_init(&twdt_config);
    if (err == ESP_ERR_INVALID_STATE) {
        ESP_LOGI("WDT", "TWDT already initialized");
    } else {
        ESP_ERROR_CHECK(err);
    }
}
 
extern "C" void app_main() {
     wifi_init_apsta();
    start_mqtt_client( );
    init_esp32_camera();
    //xTaskCreate(start_webserver, "start_webserver", 4096, NULL, 5, NULL);
    init_task_watchdog_once(); 
    
    xTaskCreate(main_task, "main_task", 4096, NULL,6, NULL);
     
}