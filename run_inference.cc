//#include "run_inference.h"
#include "model_data.h"  // TFLite 模型数据
//#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "class_prototypes.h"
//#include "sensor_module.h"  // 自定义传感器模块（相机、湿度、光感器）

//#include "tensorflow/lite/version.h"
#include <esp_task_wdt.h>
#include "esp_camera.h" 
#include "camera_config.h"
#include "classifier.h"
#include "mqtt_upload.h"
#include "esp_log.h"
#define MODEL_INPUT_SIZE 64
#define EMBEDDING_DIM 64
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




// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
 TfLiteTensor* output= nullptr;
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

constexpr int kTensorArenaSize = 350 * 1024;   
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace





// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(encoder_model_float);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    //tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
     tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will   
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
   
    //static tflite::MicroErrorReporter micro_error_reporter;
    tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddAveragePool2D(); 
    micro_op_resolver.AddReshape(); 
  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0); 
#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  esp_err_t init_status = init_esp32_camera();
  if (init_status != ESP_OK) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}




constexpr int kNumCols = 64;
constexpr int kNumRows = 64;
constexpr int kNumChannels = 1;


// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, int8_t* image_data) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

#if DISPLAY_SUPPORT
  // In case if display support is enabled, we initialise camera in rgb mode
  // Hence, we need to convert this data to grayscale to send it to tf model
  // For display we extra-polate the data to 192X192   

  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t pixel = ((uint16_t *) (fb->buf))[i * kNumCols + j];
      // for inference
      uint8_t hb = pixel & 0xFF;
      uint8_t lb = pixel >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);
      /**
       * Gamma corected rgb to greyscale formula: Y = 0.299R + 0.587G + 0.114B
       * for effiency we use some tricks on this + quantize to [-128, 127]
       */
      int8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;

      image_data[i * kNumCols + j] = grey_pixel;

      // to display
      display_buf[2 * i * kNumCols * 2 + 2 * j] = pixel;
      display_buf[2 * i * kNumCols * 2 + 2 * j + 1] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j + 1] = pixel;
    }
  }
#else // DISPLAY_SUPPORT
  MicroPrintf("Image Captured\n");
  // We have initialised camera to grayscale
  // Just quantize to int8_t
//   for (int i = 0; i < image_width * image_height; i++) {
//     image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
//   }
  const int src_w = fb->width;
  const int src_h = fb->height;
  const uint8_t* src = fb->buf;
  for (int y = 0; y < kNumRows; ++y) {
        for (int x = 0; x < kNumCols; ++x) {
            // 最近邻采样：将 160x120 → 64x64
            int src_x = x * src_w / kNumCols;
            int src_y = y * src_h / kNumRows;
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
            int input_index = y * kNumRows + x;
            image_data[input_index]  = gray / 255.0f;  // 归一化为 float32
        }
    } 
#endif // DISPLAY_SUPPORT

  esp_camera_fb_return(fb);
  /* here the esp camera can give you grayscale image directly */
  return kTfLiteOk; 
}
 

// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }
  //sensor_data_t *data = get_sensor_data(input->data.int8);

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

   output = interpreter->output(0);
   
   const float* output_data = (const float*)output->data.f;
     get_mqtt_feature(output_data);

  // Process the inference results.
//   int8_t person_score = output->data.uint8[kPersonIndex];
//   int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

//   float person_score_f =
//       (person_score - output->params.zero_point) * output->params.scale;
//   float no_person_score_f =
//       (no_person_score - output->params.zero_point) * output->params.scale;

//   // Respond to detection
//   RespondToDetection(person_score_f, no_person_score_f);
    int predicted = classifier_predict(output_data);
    printf("Predicted class: %d\n", predicted);

  vTaskDelay(1); // to avoid watchdog trigger
} 




void tensor_task(void *arg)
{
    esp_task_wdt_add(NULL);  // 注册到 watchdog
#if 0
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
    
    constexpr int kTensorArenaSize = 350 * 1024;   
    uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
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
 #else
     setup();
     while (true) {
        esp_task_wdt_reset();
        loop();
    } 
#endif

     //tensor_state =5;  // 处理完成，通知主任务可继续
    
     //ESP_LOGI(TAG, "Class Done tensor_state=%d",tensor_state);    
     
    //heap_caps_free(tensor_arena);
    esp_task_wdt_delete(NULL);  // 可选：退出前注销 watchdog
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
