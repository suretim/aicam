 
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
#include "config_mqtt.h"
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
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor= nullptr; 

  constexpr int kTensorArenaSize = 1024 * 1024;  
  static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

//float *gradients ;       // 模拟梯度

float LAMBDA_EWC = 0.001f;
float LR = 0.01f;


extern std::vector<uint8_t> ewc_buffer;
bool received_flag = false;

// 層資訊（依 Python 端 trainable_variables）
std::vector<std::vector<float>> trainable_layers;
std::vector<std::vector<float>> fisher_layers;
std::vector<std::vector<int>> layer_shapes;

  

bool ewc_assets_received() {
    return !ewc_buffer.empty(); // 可依需改成判斷完整檔案大小
}

void parse_ewc_assets() {
    // 根據 layer_shapes 切分 ewc_buffer
    size_t offset = 0;
    for(auto& shape : layer_shapes) {
        size_t len = 1;
        for(auto s : shape) len *= s;
        std::vector<float> arr(ewc_buffer.begin() + offset, ewc_buffer.begin() + offset + len);
        trainable_layers.push_back(arr);
        offset += len;
    }

    // Fisher matrix 同理（Python 端先約定順序與大小）
    for(auto& shape : layer_shapes) {
        size_t len = 1;
        for(auto s : shape) len *= s;
        std::vector<float> arr(ewc_buffer.begin() + offset, ewc_buffer.begin() + offset + len);
        fisher_layers.push_back(arr);
        offset += len;
    }

    received_flag = true;
    ESP_LOGI(TAG,"Parsed EWC assets: %zu layers", trainable_layers.size());
}




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


   // 访问输入张量和输出张量
     input_tensor  = interpreter->input_tensor(0); // 假设有一个输入张量
     output_tensor  = interpreter->output_tensor(0); // 假设有一个输出张量
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
#else
  void save_float_bin(char *path, float* data, size_t len)
  {
    // TODO: Implement saving to NVS
    return;
  }

void update_dense_layer_weights( 
                                int layer_index,
                                const std::vector<float> &weights,
                                const std::vector<int> &shape) {
    //TfLiteTensor* tensor = interpreter.tensor(layer_index);
     TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(layer_index);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
    size_t n = 1;
    for(int s : shape) n *= s;
    for(size_t i=0; i<n; ++i) tensor->data.f[i] = weights[i];
}

float compute_ewc_loss( 
                       const std::vector<std::vector<float>> &prev_weights,
                       const std::vector<std::vector<float>> &fisher_matrix) {
    float loss = 0.0f;
    for(size_t i=0; i<prev_weights.size(); ++i) {
        //TfLiteTensor* tensor = interpreter.tensor(i);
         TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(i);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
        for(size_t j=0; j<prev_weights[i].size(); ++j) {
            float diff = tensor->data.f[j] - prev_weights[i][j];
            loss += fisher_matrix[i][j] * diff * diff;
        }
    }
    return LAMBDA_EWC * loss;
}
  int update_interpreter_weights(int fisher_layer) {
    // 确保 interpreter 已初始化
    if (!interpreter) {
        printf("Interpreter not initialized!\n");
        return -1;
    }
     
    int fisher_weight_len=0;
    
    // 使用 interpreter->tensors_size() 获取张量数量
    for ( size_t i = 0 ; i < fisher_layer; ++i) {

      //TfLiteTensor* tensor = interpreter->tensor(i);
      
 
        TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(i);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
        if (tensor == nullptr) {
          printf("Tensor %d not preserved in arena\n", i);
          continue;
          //return -1;
        }
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
                size_t idx=fisher_weight_len+j;
                //float grad_ewc=  0.1f * (( rand() % 100) / 100.0f - 0.5f);
                float grad_ewc  =( 2.0f * LAMBDA_EWC * fisher_matrix[idx] * (theta[idx] - theta_old[idx])  );
                theta[idx] -= LR * grad_ewc;  // 更新权重
            }
            fisher_weight_len+=weight_len;
        }
    }
    save_float_bin("/model_weights.bin", theta, fisher_weight_len);
    return fisher_weight_len;
  }

#endif
 
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
  
    // 推理範例
    //float* input = interpreter.input(0)->data.f;
    //for(int i=0; i<SEQ_LEN*NUM_FEATS; ++i) input[i] = input_data[i];

  TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return kTfLiteError;
    }
      
    
        float* output = output_tensor->data.f;
        printf("Inference output: ");
        for(int i=0; i<NUM_CLASSES; ++i) printf("%.3f ", output[i]);
        printf("\n");

        // 計算 EWC loss
        float ewc_loss = compute_ewc_loss( trainable_layers, fisher_layers);
        printf("EWC loss: %.6f\n", ewc_loss);
    

    // int flowering = is_flowering_seq(x_input, 550.0f);
    // int toggle_flag;
    // float toggle_rate = hvac_toggle_score(x_input, 0.15f, &toggle_flag);

    // printf("Flowering: %d, Toggle Rate: %.4f, Toggle Flag: %d\n", flowering, toggle_rate, toggle_flag);
    // printf("Predicted probabilities: ");
    // for (int i=0; i<NUM_CLASSES; i++) printf("%.4f ", out_prob[i]);
    // printf("\n");

    
     get_mqtt_feature(output_tensor->data.f); 
     int predicted = classifier_predict(output_tensor->data.f);
     printf("Predicted class: %d\n", predicted); 
     vTaskDelay(1); // to avoid watchdog trigger
  return kTfLiteOk;
} 
 

// The name of this function is important for Arduino compatibility.
//TfLiteStatus setup(void) {
TfLiteStatus run_inference(float* input_seq, int seq_len, int num_feats, float* out_logits) {
    extern size_t theta_len;  
    extern size_t fisher_len; 
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    
    if(fisher_len <=0 || theta_len <= 0)
    {
      ESP_LOGE(TAG, "Failed get fisher matrix (normal if first loaded)");
      //if( safe_nvs_operation() ==0)
      //    return kTfLiteError;
    }
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
 
    tflite::MicroMutableOpResolver<24> micro_op_resolver;
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
    micro_op_resolver.AddSub();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddUnpack();  
    micro_op_resolver.AddFill();
    micro_op_resolver.AddSplit(); 
    micro_op_resolver.AddLogistic();  // This handles sigmoid activation CONCATENATION
    micro_op_resolver.AddTanh();

    micro_op_resolver.AddMean();
    micro_op_resolver.AddAbs();
    micro_op_resolver.AddConcatenation();

    static tflite::MicroInterpreter static_interpreter( model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      MicroPrintf("AllocateTensors() failed");
      return kTfLiteError;
    }
      input_tensor = interpreter->input(0);
      output_tensor = interpreter->output(0);
      if (input_tensor == nullptr || output_tensor == nullptr) {
          ESP_LOGE(TAG, "获取输入输出张量失败");
          return kTfLiteError;
      }
    // int image_width = input->dims->data[1];
    //int image_height = input->dims->data[2];
    //int channels = input->dims->data[3];
  

// 微调示意：更新权重，EWC参与
   #if 1


// 更新 Dense 層權重
    for(size_t i=0; i<trainable_layers.size(); ++i) {
        update_dense_layer_weights( i, trainable_layers[i], layer_shapes[i]);
    }

    

   #else
    int fisher_weight_len= update_interpreter_weights(FISHER_LAYER);
    if(fisher_weight_len <= 0)
    {
      return kTfLiteError;
    }

    #endif
    printf("input dims: %d %d %d %d  output dims: %d %d  \n",
        input_tensor->dims->data[0],
        input_tensor->dims->data[1],
        input_tensor->dims->data[2],
        input_tensor->dims->data[3],
        output_tensor->dims->data[0],
        output_tensor->dims->data[1] 
         );
       
        update_interpreter_weights(FISHER_LAYER);
    
   for (int t=0; t<SEQ_LEN; t++)
        for (int f=0; f<FEATURE_DIM; f++)
            input_tensor->data.f[t*FEATURE_DIM + f] = input_seq[t*FEATURE_DIM+f];
    
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
 

    // 等待完整接收
    //while(!ewc_assets_received()) {
    //    vTaskDelay(100 / portTICK_PERIOD_MS);
    //}

    // 解析 buffer -> trainable layers + Fisher matrix
    

     while (true)
     { 
        if(ewc_assets_received()==1 )
        {
            parse_ewc_assets();
        }
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
