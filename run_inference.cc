#include "run_inference.h"
#include "model_data.h"  // TFLite 模型数据
//#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "class_prototypes.h"
#include "sensor_module.h"  // 自定义传感器模块（相机、湿度、光感器）

//#include "tensorflow/lite/version.h"
#include <esp_task_wdt.h>
#include "esp_camera.h" 

#include "esp_log.h"
#define MODEL_INPUT_SIZE 64
#define EMBEDDING_DIM 64
static u_int8_t tensor_state=0;

float g_buffer[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE];
//float  feat_out[EMBEDDING_DIM ];
extern float  feat_out[EMBEDDING_DIM ];
 // extern  unsigned char g_model[];
// 设备日志
#define TENSOR_ARENA_SIZE (100 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static  camera_fb_t *fb = NULL;
typedef struct {
        int width;
        int height;
        uint8_t * data;
} fb_data_t;
static const char *TAG = "Inference";
//float g_buffer[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE];

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
 
     tensor_state =5;  // 处理完成，通知主任务可继续
    
     ESP_LOGI(TAG, "Class Done tensor_state=%d",tensor_state);    
     
    heap_caps_free(tensor_arena);
    esp_task_wdt_delete(NULL);  // 可选：退出前注销 watchdog
    vTaskDelete(NULL);  // 删除自己
    
}


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
            g_buffer[input_index]  = gray / 255.0f;  // 归一化为 float32
        }
    }  
    
}


// 运行 TFLite 推理
void run_inference(sensor_data_t data) {
    // 设置输入张量
    //float* input = ftlite_fed::interpreter->typed_input_tensor (0);
     
    //memcpy(input, data.image, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
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
    // 运行推理
    interpreter.Invoke();

    // 获取推理结果
    float* output = interpreter.typed_output_tensor<float>(0);
    float prediction = output[0];  // 假设是一个二分类问题

    ESP_LOGI(TAG, "Prediction: %f", prediction);

    // 计算不确定性（假设通过置信度来计算）
    float uncertainty = 1.0f - prediction;  // 示例：不确定性越高预测值越低
    ESP_LOGI(TAG, "Uncertainty: %f", uncertainty);

    // 基于不确定性决定是否进行本地训练（例如阈值 0.5）
    if (uncertainty > 0.5f) {
        // 进行本地训练（例如 fine-tuning）
        ESP_LOGI(TAG, "High uncertainty, training locally...");
        // 本地训练代码（此部分可以自定义）
    }
}

void run_encoder_inference(const uint8_t *input_image, float *output_vector) {


    // 初始化模型
    const tflite::Model* model = tflite::GetModel(encoder_model_float);
    // static tflite::AllOpsResolver resolver;
    // static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    // interpreter.AllocateTensors();



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

    // 获取输入 tensor 并填充数据
    TfLiteTensor* input = interpreter.input(0);  // shape: [1, 64, 64, 3]
    for (int i = 0; i < 64 * 64 * 3; ++i) {
        input->data.f[i] = (float)input_image[i] / 255.0f;
    }

    // 执行推理
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("TFLite inference failed\n");
        return;
    }

    // 获取输出 tensor（特征向量）
    TfLiteTensor* output = interpreter.output(0);  // shape: [1, 64]
    for (int i = 0; i < 64; ++i) {
        output_vector[i] = output->data.f[i];
    }
}
