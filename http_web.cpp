/**
 * This example takes a picture every 5s and print its size on serial monitor.
 */

// =============================== SETUP ======================================

// 1. Board setup (Uncomment):
// #define BOARD_WROVER_KIT
// #define BOARD_ESP32CAM_AITHINKER
// #define BOARD_ESP32S3_WROOM
// #define BOARD_ESP32S3_GOOUUU

/**
 * 2. Kconfig setup
 *
 * If you have a Kconfig file, copy the content from
 *  https://github.com/espressif/esp32-camera/blob/master/Kconfig into it.
 * In case you haven't, copy and paste this Kconfig file inside the src directory.
 * This Kconfig file has definitions that allows more control over the camera and
 * how it will be initialized.
 */

/**
 * 3. Enable PSRAM on sdkconfig:
 *
 * CONFIG_ESP32_SPIRAM_SUPPORT=y
 *
 * More info on
 * https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/kconfig.html#config-esp32-spiram-support
 */

// ================================ CODE ======================================

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



#define WIFI_SSID "1573"
#define WIFI_PASS "987654321"
#define MODEL_INPUT_SIZE 64
 
#define CAM_PIN_PWDN    -1  //power down is not used
#define CAM_PIN_RESET   -1  //software reset will be performed
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5

#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11

#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

extern const unsigned char encoder_model_float[];
extern const int encoder_model_float_len;

static const char *TAG = "example:http_web";

httpd_handle_t server = NULL;

static camera_config_t camera_config = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    //XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

                   // |--用于图像显示  PIXFORMAT_RGB888不支持
    .pixel_format = PIXFORMAT_RGB565,//PIXFORMAT_JPEG,//PIXFORMAT_RGB565, //YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_QVGA,//FRAMESIZE_QVGA,    //QQVGA-UXGA, For ESP32, do not use sizes above QVGA when not JPEG. The performance of the ESP32-S series has improved a lot, but JPEG mode always gives better frame rates.

    .jpeg_quality = 12, //0-63, for OV series camera sensors, lower number means higher quality
    .fb_count = 2,       //When jpeg mode is used, if fb_count more than one, the driver will work in continuous mode.
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

static esp_err_t init_camera(void)
{
    //initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera Init Failed");
        return err;
    }

    return ESP_OK;
}



static EventGroupHandle_t wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
 static  void wifi_event_handler0(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (id == WIFI_EVENT_AP_START) ESP_LOGI(TAG, "AP启动成功");
    else if (id == WIFI_EVENT_AP_STACONNECTED) ESP_LOGI(TAG, "设备接入");
    else if (id == WIFI_EVENT_AP_STADISCONNECTED) ESP_LOGI(TAG, "设备断开");
}
static void wifi_event_handler (void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Disconnected. Retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// ---------------- 等待 WiFi 连接 ----------------
void wait_for_wifi_connection() {
    ESP_LOGI(TAG, "Waiting for WiFi...");
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
                                           WIFI_CONNECTED_BIT,
                                           pdFALSE,
                                           pdTRUE,
                                           pdMS_TO_TICKS(15000));
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi connected.");
    } else {
        ESP_LOGE(TAG, "WiFi connection timed out.");
    }
}

void wifi_init_ap() 
{

#if 1


    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_event_group = xEventGroupCreate();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {};
    strcpy((char *)wifi_config.sta.ssid, WIFI_SSID); 
    strcpy((char *)wifi_config.sta.password, WIFI_PASS);
wifi_config.ap.ssid_len = strlen(WIFI_SSID);
wifi_config.ap.max_connection = 4;
wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

     wait_for_wifi_connection();

#else



    esp_netif_init();
    esp_event_loop_create_default();

    // 2. 注册AP事件回调（用户提供的代码）
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));

    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
wifi_config_t ap_config = { 0 };
strcpy((char *)ap_config.ap.ssid, "ESP32-CAM-AP");
strcpy((char *)ap_config.ap.password, "12345678");
ap_config.ap.ssid_len = strlen("ESP32-CAM-AP");
ap_config.ap.max_connection = 4;
ap_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    // wifi_config_t ap_config = {
    //     .ap = {
    //         .ssid = "ESP32-CAM-AP",
    //         .password = "12345678",
    //         .max_connection = 4,
    //         .authmode = WIFI_AUTH_WPA2_PSK
    //     }
    // };
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap_config);
    
    //esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G);
    //esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE);  // 指定信道6减少干扰

    esp_wifi_start();

    esp_netif_ip_info_t ip_info;
    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_AP_DEF"), &ip_info);
    ESP_LOGI(TAG, "got ip:" IPSTR "\n", IP2STR(&ip_info.ip));
    #endif
}



static esp_err_t stream_handler(httpd_req_t *req);
// 注册URI路由
httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler
};

httpd_handle_t start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if(ESP_OK != httpd_start(&server, &config))
        ESP_LOGE(TAG, "httpd_start failed");
    if(ESP_OK != httpd_register_uri_handler(server, &stream_uri))
        ESP_LOGE(TAG, "httpd_register_uri_handler failed");
    //ESP_LOGI("MAIN", "Server ready at http://192.168.4.1/stream");
    ESP_LOGI("MAIN", "Server ready at http://192.168.0.57/stream");
    
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

typedef struct {
        int width;
        int height;
        uint8_t * data;
} fb_data_t;

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
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[128];
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    size_t out_len, out_width, out_height;
    uint8_t *out_buf;
    bool s;
   const tflite::Model* model = tflite::GetModel(encoder_model_float);
//const tflite::Model* model = reinterpret_cast<const tflite::Model*>(encoder_model);
 
     
//static tflite::MicroErrorReporter micro_error_reporter;
tflite::MicroMutableOpResolver<6> resolver;
resolver.AddConv2D();
resolver.AddRelu();
resolver.AddMaxPool2D();
resolver.AddAveragePool2D(); 
resolver.AddReshape(); 
 
 
// 3. 初始化解释器（不使用 ErrorReporter）

    constexpr int kTensorArenaSize = 350 * 1024;   
    uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);


    interpreter.AllocateTensors();
    // 确保正确分配输入和输出张量


    
TfLiteTensor* input = interpreter.input(0);
if (input == nullptr) {
    ESP_LOGE(TAG, "Input tensor is null!");
    return ESP_FAIL;
}

TfLiteTensor* output = interpreter.output(0);
if (output == nullptr) {
    ESP_LOGE(TAG, "Output tensor is null!");
    return ESP_FAIL;
}
 
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
            break;
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
                fb_data_t fb_data;

                fb_data.width = fb->width;
                fb_data.height = fb->height;
                fb_data.data = rgb888_buf;

                // rectangle box
                int x, y, w, h;
                uint32_t color = 0x0000ff00;   
                
     //while (1) 
     //{
        //ESP_LOGI("MAIN", "app_main loop");
  
        // 原图像尺寸（视你配置的 frame_size 决定）
        const int src_w = fb->width;
        const int src_h = fb->height;
        const uint8_t* src = fb->buf; 
        float* input_buffer = (float*)(input->data.f);
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
                input_buffer[input_index]  = gray / 255.0f;  // 归一化为 float32
            }
        }  
        //esp_camera_fb_return(fb);
        interpreter.Invoke();
        float* feat = (float*)output->data.f ;

        float min_dist = INFINITY;
        int pred = -1;
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
        start_mqtt_client(pred,min_dist);
     //   vTaskDelay(300000 / portTICK_PERIOD_MS);
    //}



                x = 40;
                y = 40;
                w = 160;
                h = 120;
                fb_gfx_drawFastHLine(&fb_data, x, y, w, color);
                fb_gfx_drawFastHLine(&fb_data, x, y + h - 1, w, color);
                fb_gfx_drawFastVLine(&fb_data, x, y, h, color);
                fb_gfx_drawFastVLine(&fb_data, x + w - 1, y, h, color);
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
                    xTaskCreate(restart_task, "restart_task", 4096, NULL, 5, NULL);
                    return ESP_FAIL;
                }
            }
        }
        //vTaskDelay(pdMS_TO_TICKS(20));
    }
    return res;
}


extern "C" void  app_main() {
    nvs_flash_init();
    wifi_init_ap();
    init_camera();
    start_webserver();    
}
