 
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h" 
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h" 
// -------------------------
// 模型数据 (TFLite flatbuffer) lstm_encoder_contrastive 和 meta_lstm_classifier
// -------------------------
extern const unsigned char lstm_encoder_contrastive_tflite[];
extern const unsigned int lstm_encoder_contrastive_tflite_len;

extern const unsigned char meta_lstm_classifier_tflite[];
extern const unsigned int meta_lstm_classifier_tflite_len;

// -------------------------
// TensorArena
// -------------------------
#define SEQ_LEN 10
#define NUM_FEATS 64
#define NUM_CLASSES 3
static const char *TAG = "Inference_lstm";

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


// The name of this function is important for Arduino compatibility.
TfLiteStatus loop() {
  
    
  TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "模型推理失败");
        return kTfLiteError;
    }

    // get_mqtt_feature(output->data.f); 
    //int predicted = classifier_predict(output->data.f);
    //printf("Predicted class: %d\n", predicted); 
  //vTaskDelay(1); // to avoid watchdog trigger
  return kTfLiteOk;
} 
 

// The name of this function is important for Arduino compatibility.
//TfLiteStatus setup(void) {
TfLiteStatus run_inference(float* input_seq, int seq_len, int num_feats, float* out_logits) {

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.

    //encoder_model_float=asm("_binary_encoder_model_float_tflite_start"); 
    model = tflite::GetModel(lstm_encoder_contrastive_tflite);
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
 
    tflite::MicroMutableOpResolver<18> micro_op_resolver;
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddRelu(); 
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();  // 🔧 添加这个
    micro_op_resolver.AddFullyConnected();  // 如果你有 dense 层也要加
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddSoftmax();

    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddUnpack();  
    micro_op_resolver.AddFill();
    micro_op_resolver.AddSplit(); 
    micro_op_resolver.AddLogistic();  // This handles sigmoid activation


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
     
    // 7) 读取输出
     
    int num_classes = output->dims->data[1];
    memcpy(out_logits, output->data.f, num_classes * sizeof(float));
#else
  if (kTfLiteOk != loop()) {
    MicroPrintf("Image loop failed.");
    return kTfLiteError;
  } 
 
    ESP_LOGI(TAG, "推理完成，系统正常运行");
#endif
return kTfLiteOk;
}

TfLiteStatus setup(void) {
  float input_seq[SEQ_LEN * NUM_FEATS] = {25.0};  // 从传感器读取
    float logits[NUM_CLASSES];
  TfLiteStatus ret = run_inference(input_seq, SEQ_LEN, NUM_FEATS, logits);
  return ret;  
}
//u_int8_t get_tensor_state(void);
void lll_tensor_run(void) 
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
// -------------------------
// 示例
// -------------------------
//int main() {
    // 假设输入 seq_len x num_feats
    
//    tensor_run();
    //int ret = run_inference(input_seq, SEQ_LEN, NUM_FEATS, logits);
    // if (ret == 0) {
    //     printf("预测 logits: ");
    //     for (int i=0;i<NUM_CLASSES;i++) printf("%.3f ", logits[i]);
    //     printf("\n");
    // }

    //return 0;
//}
