 
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
extern bool ewc_ready;
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
  static uint8_t *tensor_arena= nullptr;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

//float *gradients ;       // 模拟梯度

float LAMBDA_EWC = 0.001f;
float LR = 0.01f;


extern std::vector<uint8_t> ewc_buffer;
bool received_flag = false;

// 層資訊（依 Python 端 trainable_variables）
extern std::vector<std::vector<float>> trainable_layers;
extern std::vector<std::vector<float>> fisher_layers;
extern std::vector<std::vector<int>> layer_shapes;
std::vector<int> trainable_tensor_indices;     // 存 dense 層的 tensor index

    
 #include "tensorflow/lite/schema/schema_generated.h"
 

// 全局變量
//trainable_tensor_indices = [0, 1, 2, 3, 6, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 50, 223, 224, 225]; 
// 將 TFLite FULLY_CONNECTED 層的 shape 提取到 layer_shapes

 

std::string shape_to_string(const std::vector<int>& shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); i++) {
        s += std::to_string(shape[i]);
        if (i < shape.size() - 1) s += ",";
    }
    return s;
}


void extract_layer_shapes_from_model(const tflite::Model* model) {
    if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) return;

    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (!subgraph) return;

    auto* operators = subgraph->operators();
    auto* tensors   = subgraph->tensors();
    if (!operators || !tensors) return;

    layer_shapes.clear();
    trainable_tensor_indices.clear();

    for (size_t op_idx = 0; op_idx < operators->size(); op_idx++) {
        const tflite::Operator* op = operators->Get(op_idx);
        if (!op) continue;

        if (!model->operator_codes() || op->opcode_index() >= model->operator_codes()->size()) continue;
        const tflite::OperatorCode* op_code = model->operator_codes()->Get(op->opcode_index());
        if (!op_code) continue;

        if (op_code->builtin_code() == tflite::BuiltinOperator_FULLY_CONNECTED) {
            // ---- Weights ----
            int weights_idx = op->inputs()->Get(1);
            if (weights_idx >= 0 && weights_idx < tensors->size()) {
                const tflite::Tensor* w = tensors->Get(weights_idx);
                if (w && w->shape()) {
                    std::string w_name = w->name() ? w->name()->str() : "unnamed";
                    if ((w_name.find("meta_dense") != std::string::npos) ||(w_name.find("hvac_dense") != std::string::npos))
                    {
                        std::vector<int> w_shape;
                        for (int d = 0; d < w->shape()->size(); d++) {
                            w_shape.push_back(w->shape()->Get(d));
                        }
                        layer_shapes.push_back(w_shape);
                        trainable_tensor_indices.push_back(weights_idx);

                        ESP_LOGI(TAG, "Dense Weights[%zu]: %s shape=[%s] -> Added idx %d",
                                 op_idx, w_name.c_str(),
                                 shape_to_string(w_shape).c_str(),
                                 weights_idx);
                    }
                }
            }

            // ---- Bias ----
            int bias_idx = (op->inputs()->size() > 2) ? op->inputs()->Get(2) : -1;
            if (bias_idx >= 0 && bias_idx < tensors->size()) {
                const tflite::Tensor* b = tensors->Get(bias_idx);
                if (b && b->shape()) {
                    std::string b_name = b->name() ? b->name()->str() : "unnamed";
                    if( (b_name.find("meta_dense") != std::string::npos)  ||(b_name.find("hvac_dense") != std::string::npos)  )
                    {
                        std::vector<int> b_shape;
                        for (int d = 0; d < b->shape()->size(); d++) {
                            b_shape.push_back(b->shape()->Get(d));
                        }
                        layer_shapes.push_back(b_shape);
                        trainable_tensor_indices.push_back(bias_idx);

                        ESP_LOGI(TAG, "Dense Bias[%zu]: %s shape=[%s] -> Added idx %d",
                                 op_idx, b_name.c_str(),
                                 shape_to_string(b_shape).c_str(),
                                 bias_idx);
                    }
                }
            }
        }
    }

    ESP_LOGI(TAG, "Extracted %zu dense layer tensors into layer_shapes, %zu trainable indices",
             layer_shapes.size(), trainable_tensor_indices.size());
}

 
void update_dense_layer_weights(void)
{
    extern std::vector<std::vector<float>> trainable_layers;
    extern std::vector<std::vector<float>> fisher_layers;
    extern std::vector<std::vector<int>> layer_shapes;
    extern std::vector<int> trainable_tensor_indices;  // 建議新增，存放哪些 tensor 是 trainable

    for (size_t k = 0; k < trainable_tensor_indices.size(); ++k) {
        int j = trainable_tensor_indices[k];   // 取得對應 tensor id
        TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(j);
        TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(eval_tensor);
        if (!tensor || tensor->type != kTfLiteFloat32 || tensor->dims->size < 1)
            continue;

        float* theta = tensor->data.f;   // 當前權重
        std::vector<float>& theta_star = trainable_layers[k];  // 舊任務權重
        std::vector<float>& fisher = fisher_layers[k];         // Fisher
        std::vector<int>& layer_shape = layer_shapes[k];

        // 計算該層權重總數
        size_t n = 1;
        for (int s : layer_shape) n *= s;
        printf("Layer %d train weights nums: %zu\n", j, n);

        // EWC 更新
        for (size_t i = 0; i < n; ++i) {
            float grad_ewc = 2.0f * LAMBDA_EWC * fisher[i] * (theta[i] - theta_star[i]);
            theta[i] -= LR * grad_ewc;   // 直接更新 interpreter tensor
        }
    }
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
 

void parse_ewc_assets() {
    if (!ewc_ready || ewc_buffer.empty()) return;

    extract_layer_shapes_from_model(model);
    trainable_layers.clear();
    fisher_layers.clear();

    size_t offset = 0;
    
    // Trainable layers
    for (size_t i = 0; i < layer_shapes.size(); ++i) {
        size_t len = 1;
        for (auto s : layer_shapes[i]) len *= s;

        if (offset + len > ewc_buffer.size()) {
            ESP_LOGE("EWC", "Not enough data for trainable layer %zu", i);
            return; // 避免越界
        }

        std::vector<float> layer_data(len);
        memcpy(layer_data.data(), ewc_buffer.data() + offset, len * sizeof(float));
        trainable_layers.push_back(layer_data);
        offset += len;

        ESP_LOGI("EWC", "Parsed trainable layer %zu, len=%zu", i, len);
    }

    // Fisher layers
    for (size_t i = 0; i < layer_shapes.size(); ++i) {
        size_t len = 1;
        for (auto s : layer_shapes[i]) len *= s;

        if (offset + len > ewc_buffer.size()) {
            ESP_LOGE("EWC", "Not enough data for fisher layer %zu", i);
            return; // 避免越界
        }

        std::vector<float> arr(
            ewc_buffer.begin() + offset,
            ewc_buffer.begin() + offset + len
        );
        fisher_layers.push_back(std::move(arr));
        offset += len;
    }

    ESP_LOGI("EWC", "Parsed EWC assets: %zu trainable, %zu fisher layers",
             trainable_layers.size(), fisher_layers.size());

    // 用完清空 buffer
    ewc_buffer.clear();
    
    if (!trainable_layers.empty()) { 
        update_dense_layer_weights();
        

        trainable_layers.clear();  // 可選，保留 capacity
        fisher_layers.clear();
        ESP_LOGI("Main", "All layers updated");
    }
    ewc_ready = false;
}


  
// The name of this function is important for Arduino compatibility.
//TfLiteStatus setup(void) {
TfLiteStatus run_inference(float* input_seq, int seq_len, int num_feats, float* out_logits) {
     
    model = tflite::GetModel(meta_lstm_classifier_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      MicroPrintf("Model provided is schema version %d not equal to supported "
                  "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
      return kTfLiteError ;
    } 
    //if(ewc_assets_received()==1 )
    
    //safe_analyze_model(model);
    
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
      ESP_LOGE(TAG, "AllocateTensors() failed");
      return kTfLiteError;
    }
      input_tensor = interpreter->input(0);
      output_tensor = interpreter->output(0);
      if (input_tensor == nullptr || output_tensor == nullptr) {
          ESP_LOGE(TAG, "failed to get input or output tensor");
          return kTfLiteError;
      }
     
    // 微调示意：更新权重，EWC参与 
    parse_ewc_assets();   


     
    printf("input dims: %d %d %d %d  output dims: %d %d  \n",
        input_tensor->dims->data[0],
        input_tensor->dims->data[1],
        input_tensor->dims->data[2],
        input_tensor->dims->data[3],
        output_tensor->dims->data[0],
        output_tensor->dims->data[1] 
         );
       
        
   for (int t=0; t<SEQ_LEN; t++)
        for (int f=0; f<FEATURE_DIM; f++)
            input_tensor->data.f[t*FEATURE_DIM + f] = input_seq[t*FEATURE_DIM+f];
     
    
 //ESP_LOGI(TAG, "模型输入大小: %d", input->bytes);
     
    // 7) 读取输出
     
    //int num_classes = output->dims->data[1];
    //memcpy(out_logits, output->data.f, num_classes * sizeof(float));
 
  if (kTfLiteOk != loop()) {
    MicroPrintf(" inference loop failed.");
    return kTfLiteError;
  } 
   //   interpreter->ResetTempAllocations();

    //free(tensor_arena);
   // ESP_LOGI(TAG, "推理完成，系统正常运行");
 
return kTfLiteOk;
}


float input_seq[SEQ_LEN * FEATURE_DIM] = {25.0};  // 从传感器读取
float logits[NUM_CLASSES];

 
//u_int8_t get_tensor_state(void);
void lll_tensor_run(void) 
{
     

     while (true)
     { 
        
        if(kTfLiteError ==  run_inference(input_seq, SEQ_LEN, FEATURE_DIM, logits) )
        {
          break; 
        }
          
      vTaskDelay(pdMS_TO_TICKS(60000));  // 每60秒输出一次
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
