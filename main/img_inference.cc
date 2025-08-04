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
#define MODEL_INPUT_SIZE 64
#define EMBEDDING_DIM 64

constexpr int kNumCols = 64;
constexpr int kNumRows = 64;
constexpr int kNumChannels = 3;
//static u_int8_t tensor_state=0;

//float g_buffer[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE];
//float  feat_out[EMBEDDING_DIM ];
//extern float  feat_out[EMBEDDING_DIM ];
 // extern  unsigned char g_model[];
// 设备日志
//#define TENSOR_ARENA_SIZE (100 * 1024)
//static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// static  camera_fb_t *fb_pic = NULL;
// typedef struct {
//         int width;
//         int height;
//         uint8_t * data;
// } fb_data_t;
static const char *TAG = "Inference";


 //extern      SemaphoreHandle_t web_send_mutex;


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
 TfLiteTensor* output= nullptr;
   float* output_data=nullptr;
   float* input_data=nullptr;
// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#if CONFIG_NN_OPTIMIZED
constexpr int scratchBufSize = 60 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
// Keeping allocation on bit larger size to accomodate future needs.
//constexpr int kTensorArenaSize = 100 * 1024 + scratchBufSize;

constexpr int kTensorArenaSize = 1024 * 1024;  
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace


void reset_tensor(void)
{
  //free(tensor_arena);
  heap_caps_free(tensor_arena);
}

extern const unsigned char encoder_model_float[] asm("_binary_encoder_model_tflite_start");

// The name of this function is important for Arduino compatibility.
TfLiteStatus setup() {
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
      //tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      
      tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    }
    if (tensor_arena == NULL) {
      printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
      return kTfLiteError;
    }
 
    tflite::MicroMutableOpResolver<12> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();  // 🔧 添加这个
    micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMul();
    
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return kTfLiteError;
  }
  //input = interpreter->input(0);
    output = interpreter->output(0);
    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "获取输入输出张量失败");
        return kTfLiteError;
    }
  // Get information about the memory area to use for the model's input.
   
    input_data = (  float*)output->data.f;
    output_data = (  float*)output->data.f;
#if 0    
 ESP_LOGI(TAG, "模型输入大小: %d", input->bytes);

if (kTfLiteOk != GetESPImage(kNumCols, kNumRows, kNumChannels,input->data.f)) {
    MicroPrintf("Image capture failed.");
    return;
  }
    // 11. 模拟输入数据，确保数据符合模型输入格式
    // 这里假设输入是 float32，大小等于 input->bytes / sizeof(float)
    // float *input_buffer = input->data.f;
    // for (int i = 0; i < input->bytes / sizeof(float); i++) {
    //     input_buffer[i] = 0.0f;  // 示例全部清零
    // }
// 12. 推理执行
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "模型推理失败");
        return;
    }

    // 13. 读取输出结果
    float *output_buffer = output->data.f;
    int output_size = output->bytes / sizeof(float);

    // ESP_LOGI(TAG, "推理结果：");
    // for (int i = 0; i < output_size; i++) {
    //     ESP_LOGI(TAG, "  output[%d] = %f", i, output_buffer[i]);
    // }

    ESP_LOGI(TAG, "推理完成，系统正常运行");
#endif
return kTfLiteOk;
}


// The name of this function is important for Arduino compatibility.
TfLiteStatus loop() {
     
  if (kTfLiteOk != GetESPImage(kNumCols, kNumRows, kNumChannels,input->data.f)) {
    MicroPrintf("Image capture failed.");
    return kTfLiteError;
  }
  TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "模型推理失败");
        return kTfLiteError;
    }

     get_mqtt_feature(output_data); 
    int predicted = classifier_predict(output_data);
    printf("Predicted class: %d\n", predicted); 
  //vTaskDelay(1); // to avoid watchdog trigger
  return kTfLiteOk;
} 

u_int8_t get_tensor_state(void);
void tensor_server(void) 
{

     //setup();
     while (true)
     {
      if(kTfLiteError== setup())
               { 
                  break;
               }
      //if (xSemaphoreTake(web_send_mutex, portMAX_DELAY) == pdTRUE) {
          //if(get_tensor_state()){
            if(kTfLiteError== loop())
            {
              reset_tensor();
              break; 
            }
         // }
      //    xSemaphoreGive(web_send_mutex);
      //}
      // if(get_tensor_state()){
      //   loop();
      // }
      vTaskDelay(pdMS_TO_TICKS(10000));  // 每10秒输出一次
      //  vTaskDelay(30000 / portTICK_PERIOD_MS);
    }  
    reset_tensor();
}
void tensor_task(void *arg)
{
   // esp_task_wdt_add(NULL);  // 注册到 watchdog 
     setup();
     while (true) {
        //esp_task_wdt_reset();
        loop();
    }  
     //tensor_state =5;  // 处理完成，通知主任务可继续
    
     //ESP_LOGI(TAG, "Class Done tensor_state=%d",tensor_state);    
     
    //heap_caps_free(tensor_arena);
    //esp_task_wdt_delete(NULL);  // 可选：退出前注销 watchdog
    vTaskDelete(NULL);  // 删除自己
    
}
 
// void run_encoder_inference(const uint8_t *input_image, float *output_vector) {


//     // 初始化模型
//     const tflite::Model* model = tflite::GetModel(encoder_model_float);
//     // static tflite::AllOpsResolver resolver;
//     // static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
//     // interpreter.AllocateTensors();



// //static tflite::MicroErrorReporter micro_error_reporter;
//     tflite::MicroMutableOpResolver<6> resolver;
//     resolver.AddConv2D();
//     resolver.AddRelu();
//     resolver.AddMaxPool2D();
//     resolver.AddAveragePool2D(); 
//     resolver.AddReshape(); 
    
//     constexpr int kTensorArenaSize = 350 * 1024;   
//     uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
//     tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
 
//     interpreter.AllocateTensors();

//     // 获取输入 tensor 并填充数据
//     TfLiteTensor* input = interpreter.input(0);  // shape: [1, 64, 64, 3]
//     for (int i = 0; i < 64 * 64 * 3; ++i) {
//         input->data.f[i] = (float)input_image[i] / 255.0f;
//     }

//     // 执行推理
//     TfLiteStatus invoke_status = interpreter.Invoke();
//     if (invoke_status != kTfLiteOk) {
//         printf("TFLite inference failed\n");
//         return;
//     }

//     // 获取输出 tensor（特征向量）
//     TfLiteTensor* output = interpreter.output(0);  // shape: [1, 64]
//     for (int i = 0; i < 64; ++i) {
//         output_vector[i] = output->data.f[i];
//     }
// }
