#include <esp_log.h>
#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h>
#include <string.h>
#include "esp_camera.h" 

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  


//#include <esp_system.h>
static const char *TAG = "esp_camera";
 #define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
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
#include "esp_camera.h"
 

static camera_config_t camera_config  = {
    .pin_pwdn       = CAM_PIN_PWDN,
    .pin_reset      = CAM_PIN_RESET,
    .pin_xclk       = CAM_PIN_XCLK,
    .pin_sccb_sda   = CAM_PIN_SIOD,
    .pin_sccb_scl   = CAM_PIN_SIOC,
    .pin_d7         = CAM_PIN_D7,
    .pin_d6         = CAM_PIN_D6,
    .pin_d5         = CAM_PIN_D5,
    .pin_d4         = CAM_PIN_D4,
    .pin_d3         = CAM_PIN_D3,
    .pin_d2         = CAM_PIN_D2,
    .pin_d1         = CAM_PIN_D1,
    .pin_d0         = CAM_PIN_D0,
    .pin_vsync      = CAM_PIN_VSYNC,
    .pin_href       = CAM_PIN_HREF,
    .pin_pclk       = CAM_PIN_PCLK,
    .xclk_freq_hz   = 10000000,
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,
    .pixel_format   = PIXFORMAT_RGB565,   // JPEG or RGB565
    .frame_size     = FRAMESIZE_QVGA, //FRAMESIZE_QVGA,     // QQVGA / QVGA
    .jpeg_quality   = 12,
    .fb_count       = 2,                  // 单帧缓冲，稳定
    .fb_location    = CAMERA_FB_IN_PSRAM,  // 避免 PSRAM 问题
    .grab_mode      = CAMERA_GRAB_LATEST, // 优先取最新帧，避免队列堆积
};

//.fb_location = CAMERA_FB_IN_DRAM,   // 而不是 CAMERA_FB_IN_PSRAM
//.fb_count = 1,                      // 只用一帧缓存，避免超限

esp_err_t init_esp32_camera(void)
{
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera Init Failed: %s", esp_err_to_name(err));
        return err;
    }
    size_t free_heap = esp_get_free_heap_size();
    printf("Free heap memory: %d bytes\n", free_heap);
    // sensor_t *s = esp_camera_sensor_get();
    // if (s != NULL)
    // {
    //     ESP_LOGI(TAG, "Sensor PID: 0x%x", s->id.PID);
    //     // 可选：设置亮度/对比度等
    //     s->set_brightness(s, 1);     // -2 to 2
    //     s->set_contrast(s, 1);       // -2 to 2
    // }

    return ESP_OK;
}


