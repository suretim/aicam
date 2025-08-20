 
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "esp_log.h"
#include "esp_system.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h" 
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h" 
#include "classifier_storage.h"
#include "classifier.h"
// -------------------------
// 模型数据 (TFLite flatbuffer) lstm_encoder_contrastive 和 meta_lstm_classifier
// -------------------------
//extern const unsigned char lstm_encoder_contrastive_tflite[];
//extern const unsigned int lstm_encoder_contrastive_tflite_len;

extern const unsigned char meta_lstm_classifier_tflite[];
extern const unsigned int meta_lstm_classifier_tflite_len;

extern float *fisher_matrix;
extern float *theta ; 

// -------------------------
// TensorArena
// -------------------------

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

//float *gradients ;       // 模拟梯度

float LAMBDA_EWC = 0.001f;
float LR = 0.01f;
size_t theta_len=0;
size_t fisher_len=0;
size_t fisher_vi_len=0;


#define SPIFFS_FL      0
#define NVS_FL        1
#define FLTYPE        NVS_FL

#if FLTYPE   ==     SPIFFS_FL
#include <spiffs.h>
void spiffs_init(){
  if (!SPIFFS.begin(true)) {
        //Serial.println("SPIFFS 初始化失败！");
        return;
    }
}
void spiffs_free(){
   SPIFFS.end(); // 关闭 SPIFFS
   return;
}
// ----------------------------
// 读取 float 二进制文件
// ----------------------------
float* load_float_bin(const char* path, size_t& out_len){
    File f = SPIFFS.open(path, FILE_READ);
    if(!f){
        Serial.print("File not found: "); Serial.println(path);
        out_len = 0;
        return nullptr;
    }
    size_t size_bytes = f.size();
    out_len = size_bytes / sizeof(float);
    float* data = new float[out_len];
    f.read((uint8_t*)data, size_bytes);
    f.close();
    return data;
} 



bool load_fisher_matrix(){
    
    if(!SPIFFS.begin(true)){
        Serial.println("SPIFFS mount failed");
        return;
    }

    // 1️⃣ 加载权重和 Fisher 矩阵
    theta = load_float_bin("/model_weights.bin", theta_len);
    fisher_matrix = load_float_bin("/fisher_matrix.bin", fisher_len);
    if(!theta || !fisher_matrix || fisher_len != theta_len){
      fisher_vi_len=0;
        Serial.println("Load error or length mismatch");
        if(theta) delete[] theta;
        if(fisher_matrix) delete[] fisher_matrix;
        return false;
    }
    fisher_vi_len=fisher_len/theta_len;
    Serial.println("fisher_vi_len = %d",fisher_vi_len);
    // 保存旧权重
    theta_old = new float[theta_len];
    memcpy(theta_old, theta, theta_len*sizeof(float));
  }
#else
  void save_float_bin(char *path, float* data, size_t len)
  {
    // TODO: Implement saving to NVS
    return;
  }

 void load_fisher_matrix(int fisher_layer) {
    // 确保 interpreter 已初始化
    if (!interpreter) {
        printf("Interpreter not initialized!\n");
        return;
    }
    
    // 访问输入张量和输出张量
    TfLiteTensor* input_tensor = interpreter->input_tensor(0); // 假设有一个输入张量
    TfLiteTensor* output_tensor = interpreter->output_tensor(0); // 假设有一个输出张量
 // 打印张量信息
    printf("Input tensor shape: ");
    for (size_t j = 0; j < input_tensor->dims->size; ++j) {
        printf("%d ", input_tensor->dims->data[j]);
    }
    printf("\n");

    printf("Output tensor shape: ");
    for (size_t j = 0; j < output_tensor->dims->size; ++j) {
        printf("%d ", output_tensor->dims->data[j]);
    }
    printf("\n");
    // 使用 interpreter->tensors_size() 获取张量数量
    for (size_t i = 0; i < fisher_layer; ++i) {
        //TfLiteTensor* tensor = interpreter->tensor(i);
        TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(i);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
        //float* tensor_data = tensor->data.f;  // Access the tensor data (float32 values)
        float* theta_old = tensor->data.f;
        if (tensor->type == kTfLiteFloat32 && tensor->dims->size > 1) {
            printf("Tensor index: %zu, Tensor shape: ", i);
            for (size_t j = 0; j < tensor->dims->size; ++j) {
                printf("%d ", tensor->dims->data[j]);
            }
            printf("\n");

            size_t weight_len = 1;
            for (size_t j = 0; j < tensor->dims->size; ++j) {
                weight_len *= tensor->dims->data[j];
            }

            

            for (size_t j = 0; j < weight_len; ++j) {
                float grad_ewc = 2.0f * LAMBDA_EWC * fisher_matrix[j] * (theta[j] - theta_old[j]);
                theta[j] -= LR * grad_ewc;  // 更新权重
            }
        }
    }
    return;
}

#endif
 
 
// ----------------------------
// 更新权重到 Interpreter tensor
// ----------------------------
void update_interpreter_weights(void) {
    TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(1);
    TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
    float* tensor_data = tensor->data.f;  // Access the tensor data (float32 values)
 
    // 5️⃣ 计算梯度 & EWC 微调
    float* grad = new float[theta_len];

    //srandom(esp_timer_get_time());  // Use system time as the seed
    
   
    for (size_t i = 0; i < theta_len; i++) {
        // Generate random float between -0.05 and 0.05
        //grad[i] = 0.1f * ((esp_random() % 100) / 100.0f - 0.5f);
        grad[i] = 0.1f;
    } 

    
   
   for (int i = 0; i < theta_len; i++) {
       for (int j = 0; j < fisher_vi_len; j++) {
           // 更新权重： w = w - lr*grad + lambda*F*(w - w_prev)
           float delta = LR * grad[i] + 
               LAMBDA_EWC * fisher_matrix[i * fisher_vi_len + j] * 
               (tensor_data[i * fisher_vi_len + j] - theta_old[i * fisher_vi_len + j]);
           tensor_data[i * fisher_vi_len + j] -= delta;
       }
   } 

   save_float_bin("/model_weights.bin", theta, theta_len);
   
   // 清理
  //  delete[] theta;
  //  delete[] theta_old;
  //  delete[] fisher_matrix;
   delete[] grad;
  }
// ---------------------------
// Flowering/HVAC 判定
// ---------------------------
int is_flowering_seq(float x_seq[SEQ_LEN][FEATURE_DIM], float th_light)
{
    float mean_light = 0.0f;
    for (int t=0; t<SEQ_LEN; t++) mean_light += x_seq[t][2];
    mean_light /= SEQ_LEN;
    return mean_light >= th_light;
}
float hvac_toggle_score(float x_seq[SEQ_LEN][FEATURE_DIM], float th_toggle, int *flag) {
    float diff_sum = 0.0f;
    int count = 0;
    for (int t=1; t<SEQ_LEN; t++)
        for (int f=3; f<7; f++) {
            diff_sum += fabsf(x_seq[t][f] - x_seq[t-1][f]);
            count++;
        }
    float rate = diff_sum / count;
    *flag = rate >= th_toggle;
    return rate;
}


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
    // int flowering = is_flowering_seq(x_input, 550.0f);
    // int toggle_flag;
    // float toggle_rate = hvac_toggle_score(x_input, 0.15f, &toggle_flag);

    // printf("Flowering: %d, Toggle Rate: %.4f, Toggle Flag: %d\n", flowering, toggle_rate, toggle_flag);
    // printf("Predicted probabilities: ");
    // for (int i=0; i<NUM_CLASSES; i++) printf("%.4f ", out_prob[i]);
    // printf("\n");

    
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
    model = tflite::GetModel(meta_lstm_classifier_tflite);
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
 
    tflite::MicroMutableOpResolver<20> micro_op_resolver;
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
micro_op_resolver.AddTanh();

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
   // int image_width = input->dims->data[1];
  //int image_height = input->dims->data[2];
  //int channels = input->dims->data[3];
  printf("input dims: %d %d %d %d  output dims: %d %d \n",
       input->dims->data[0],
       input->dims->data[1],
       input->dims->data[2],
       input->dims->data[3],
       output->dims->data[0],
       output->dims->data[1]);

// 微调示意：更新权重，EWC参与
   
    if(theta_len > 0)
    {
       load_fisher_matrix(input->dims->data[3]);
       //update_interpreter_weights();
    }
   for (int t=0; t<SEQ_LEN; t++)
        for (int f=0; f<FEATURE_DIM; f++)
            input->data.f[t*FEATURE_DIM + f] = input_seq[t*FEATURE_DIM+f];
    
  // Get information about the memory area to use for the model's input.
   
    // input_data = (  float*)input->data.f;
    // output_data = (  float*)output->data.f;
    
 //ESP_LOGI(TAG, "模型输入大小: %d", input->bytes);
     
    // 7) 读取输出
     
    //int num_classes = output->dims->data[1];
    //memcpy(out_logits, output->data.f, num_classes * sizeof(float));
 
  if (kTfLiteOk != loop()) {
    MicroPrintf("Image loop failed.");
    return kTfLiteError;
  } 
 
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
return kTfLiteOk;
}


float input_seq[SEQ_LEN * FEATURE_DIM] = {25.0};  // 从传感器读取
float logits[NUM_CLASSES];


TfLiteStatus setup(void) {
 
  TfLiteStatus ret = run_inference(input_seq, SEQ_LEN, FEATURE_DIM, logits);
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
