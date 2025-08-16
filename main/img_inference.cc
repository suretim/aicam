#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h" 
//#include "sensor_module.h"  // 自定义传感器模块（相机、湿度、光感器）

#include <esp_task_wdt.h>
#include "esp_camera.h" 
#include "config_camera.h"
#include "classifier.h"
#include "config_mqtt.h"
#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h" 
#include "freertos/semphr.h"  // 这是定义信号量相关函数和类型的头文件
 

#define RGB565          0
#define RGB888        1
#define RGBTYPE        RGB565
 
static const char *TAG = "Inference";


 //extern      SemaphoreHandle_t web_send_mutex;


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
 TfLiteTensor* output= nullptr; 

constexpr int kTensorArenaSize = 1024 * 1024;  
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace


void reset_tensor(void)
{
  //free(tensor_arena);
  heap_caps_free(tensor_arena);
}

 //extern      SemaphoreHandle_t web_send_mutex;
// Get an image from the camera module
TfLiteStatus GetESPImage(void) {
  
  int image_width = input->dims->data[1];
  int image_height = input->dims->data[2];
  int channels = input->dims->data[3];
  printf("input dims: %d %d %d %d\n",
       input->dims->data[0],
       input->dims->data[1],
       input->dims->data[2],
       input->dims->data[3]);
    //if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE){
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            return kTfLiteError;
        }  
    //     xSemaphoreGive(web_send_mutex);
    //}
#if RGBTYPE   ==     RGB888

    uint8_t *rgb888_buf = (uint8_t *)heap_caps_malloc(fb->width * fb->height * 3, MALLOC_CAP_SPIRAM);
    if(rgb888_buf == NULL)
    {
        esp_camera_fb_return(fb); 
        ESP_LOGE(TAG, "rgb888_buf failed " );
        return kTfLiteError;
    }
    if(fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf) != true)
    {
        ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
        esp_camera_fb_return(fb);
        free(rgb888_buf);
        return kTfLiteError;
    }
#endif
  
    const int src_w = fb->width;
    const int src_h = fb->height;
    const int src_channel=2;
    const uint8_t* src =fb->buf;// rgb888_buf;//fb->buf;
    printf("input dims: %d %d %d and src dims %d %d %d\n",
       input->dims->data[1],input->dims->data[2],input->dims->data[3],src_w,src_w,src_channel);
    for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
            // 最近邻采样：将 160x120 → 64x64
            int src_x = x * src_w / image_width;
            int src_y = y * src_h / image_height;
            int index = (src_y * src_w + src_x) * src_channel; // 每像素2字节(RGB565)

            // 提取 RGB565 中的灰度近似
            uint8_t byte1 = src[index];
            uint8_t byte2 = src[index + 1];

            // 解码 RGB565 → 灰度
            uint8_t r = (byte2 & 0xF8);
            uint8_t g = ((byte2 & 0x07) << 5) | ((byte1 & 0xE0) >> 3);
            uint8_t b = (byte1 & 0x1F) << 3;
            //float gray = (r * 30 + g * 59 + b * 11) / 100;

            // 写入模型输入张量
            int input_index = y * image_height + x;
            input->data.f[input_index*channels+0]  = r * 30.0f;  // 归一化为 float32
            input->data.f[input_index*channels+1]  = g * 59.0f;  // 归一化为 float32
            input->data.f[input_index*channels+2]  = b * 11.0f;  // 归一化为 float32
        }
    }   
     
    esp_camera_fb_return(fb);
#if RGBTYPE   ==     RGB888
    free(rgb888_buf);  
#endif
    size_t free_heap = esp_get_free_heap_size();
    printf("Free heap memory: %d bytes\n", free_heap);
  /* here the esp camera can give you grayscale image directly */
  return kTfLiteOk; 
}
 
 
// The name of this function is important for Arduino compatibility.
TfLiteStatus loop() {
  
   
  if (kTfLiteOk != GetESPImage()) {
    MicroPrintf("Image capture failed.");
    return kTfLiteError;
  }
  TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "模型推理失败");
        return kTfLiteError;
    }

     get_mqtt_feature(output->data.f); 
    int predicted = classifier_predict(output->data.f);
    printf("Predicted class: %d\n", predicted); 
  //vTaskDelay(1); // to avoid watchdog trigger
  return kTfLiteOk;
} 
 

extern const unsigned char encoder_model_float[]  asm("_binary_student_fp32_tflite_start");
 
// The name of this function is important for Arduino compatibility.
TfLiteStatus setup(void) {
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.

    //encoder_model_float=asm("_binary_encoder_model_float_tflite_start"); 
    model = tflite::GetModel(encoder_model_float);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      MicroPrintf("Model provided is schema version %d not equal to supported "
                  "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
      return kTfLiteError ;
    }

    if (tensor_arena == NULL) {
       tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    }
    if (tensor_arena == NULL) {
      printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
      return kTfLiteError;
    }
 
    tflite::MicroMutableOpResolver<11> micro_op_resolver;
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddRelu();
    //micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();  // 🔧 添加这个
    micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddSoftmax();
    //micro_op_resolver.AddAdd();
    //micro_op_resolver.AddMul();
    micro_op_resolver.AddShape();
    
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return kTfLiteError;
  }
    input = interpreter->input(0);
    output = interpreter->output(0);
    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "获取输入输出张量失败");
        return kTfLiteError;
    }
  // Get information about the memory area to use for the model's input.
   
    // input_data = (  float*)input->data.f;
    // output_data = (  float*)output->data.f;
#if 1   
 //ESP_LOGI(TAG, "模型输入大小: %d", input->bytes);

  if (kTfLiteOk != loop()) {
    MicroPrintf("Image loop failed.");
    return kTfLiteError;
  } 
 
    ESP_LOGI(TAG, "推理完成，系统正常运行");
#endif
return kTfLiteOk;
}


//u_int8_t get_tensor_state(void);
void tensor_run(void) 
{
 
     while (true)
     {
     
       
            if(kTfLiteError== setup())
            {
              break; 
            }
          
      vTaskDelay(pdMS_TO_TICKS(10000));  // 每10秒输出一次
      //  vTaskDelay(30000 / portTICK_PERIOD_MS);
    }  
    reset_tensor();
}
void tensor_task(void *arg)
{
    esp_task_wdt_add(NULL);  // 注册到 watchdog 
   
     while (true) {
        if(kTfLiteError== setup())
            {
              break; 
            }
            vTaskDelay(pdMS_TO_TICKS(10000));  
    }  
     //tensor_state =5;  // 处理完成，通知主任务可继续
    
     //ESP_LOGI(TAG, "Class Done tensor_state=%d",tensor_state);    
     
    //heap_caps_free(tensor_arena);
    //esp_task_wdt_delete(NULL);  // 可选：退出前注销 watchdog
    reset_tensor();
    vTaskDelete(NULL);  // 删除自己
   
}
 

int run_encoder(const float* image_data, float* feature_out) {
    // Copy input
    int n = input_tensor->bytes / sizeof(float);
    for (int i = 0; i < n; i++) input_tensor->data.f[i] = image_data[i];

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke failed\n");
        return -1;
    }
    // output_tensor->data.f length should match FEATURE_DIM
    int out_n = output_tensor->bytes / sizeof(float);
    if (out_n != FEATURE_DIM) {
        printf("Warning: output feature dim mismatch: %d vs %d\n", out_n, FEATURE_DIM);
    }
    for (int i = 0; i < FEATURE_DIM && i < out_n; i++) {
        feature_out[i] = output_tensor->data.f[i];
    }
    return 0;
}
