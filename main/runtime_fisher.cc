#include <Arduino.h>
#include <SPIFFS.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

const int tensor_arena_size = 10*1024;
uint8_t tensor_arena[tensor_arena_size];

// ----------------------------
// 读取 TFLite FlatBuffer
// ----------------------------
uint8_t* load_tflite_model(const char* path, size_t& model_size){
    File f = SPIFFS.open(path, FILE_READ);
    if(!f){
        Serial.print("TFLite model not found: "); Serial.println(path);
        model_size = 0;
        return nullptr;
    }
    model_size = f.size();
    uint8_t* buffer = new uint8_t[model_size];
    f.read(buffer, model_size);
    f.close();
    return buffer;
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

// ----------------------------
// 保存 float 数组到 SPIFFS
// ----------------------------
bool save_float_bin(const char* path, float* data, size_t len){
    File f = SPIFFS.open(path, FILE_WRITE);
    if(!f) return false;
    f.write((uint8_t*)data, len*sizeof(float));
    f.close();
    return true;
}

// ----------------------------
// 模拟梯度计算（可替换为实际 loss 梯度）
// ----------------------------
void compute_dummy_gradient(float* grad, size_t len){
    for(size_t i=0;i<len;i++){
        grad[i] = 0.1f * (random(0,100)/100.0f - 0.5f);
    }
}

// ----------------------------
// EWC 更新函数
// ----------------------------
void ewc_update(float* theta, const float* theta_old, const float* fisher_matrix, const float* grad, size_t len, float lr, float lambda_ewc){
    for(size_t i=0;i<len;i++){
        float grad_ewc = 2.0f * lambda_ewc * fisher_matrix[i] * (theta[i] - theta_old[i]);
        theta[i] -= lr * (grad[i] + grad_ewc);
    }
}

// ----------------------------
// TFLite 推理（输入/输出 float）
// ----------------------------
float tflite_inference(float* input_data, int input_len, tflite::MicroInterpreter* interpreter){
    TfLiteTensor* input = interpreter->input(0);
    for(int i=0;i<input_len;i++){
        input->data.f[i] = input_data[i];
    }

    if(interpreter->Invoke() != kTfLiteOk){
        Serial.println("Invoke failed");
        return NAN;
    }

    TfLiteTensor* output = interpreter->output(0);
    return output->data.f[0];
}

// ----------------------------
// 更新权重到 Interpreter tensor
// ----------------------------
void update_interpreter_weights(tflite::MicroInterpreter* interpreter, float* theta, int len, int weight_tensor_index){
    TfLiteTensor* tensor = interpreter->tensor(weight_tensor_index);
    for(int i=0;i<len;i++){
        tensor->data.f[i] = theta[i];
    }
}

// ----------------------------
// 主循环逻辑
// ----------------------------
void run_real_time_loop(){
    size_t theta_len, fisher_len;

    if(!SPIFFS.begin(true)){
        Serial.println("SPIFFS mount failed");
        return;
    }

    // 1️⃣ 加载权重和 Fisher 矩阵
    float* theta = load_float_bin("/model_weights.bin", theta_len);
    float* fisher_matrix = load_float_bin("/fisher_matrix.bin", fisher_len);
    if(!theta || !fisher_matrix || fisher_len != theta_len){
        Serial.println("Load error or length mismatch");
        if(theta) delete[] theta;
        if(fisher_matrix) delete[] fisher_matrix;
        return;
    }

    // 保存旧权重
    float* theta_old = new float[theta_len];
    memcpy(theta_old, theta, theta_len*sizeof(float));

    // 2️⃣ 加载 TFLite 模型
    size_t model_size;
    uint8_t* model_buffer = load_tflite_model("/meta_lstm_classifier.tflite", model_size);
    if(!model_buffer){
        delete[] theta; delete[] theta_old; delete[] fisher_matrix;
        return;
    }

    tflite::MicroMutableOpResolver<10> resolver;
    tflite::MicroInterpreter interpreter((const tflite::Model*)model_buffer,
                                         resolver, tensor_arena, tensor_arena_size, nullptr);
    interpreter.AllocateTensors();

    // 假设 weight tensor 索引为 1（实际需查模型）
    int weight_tensor_index = 1;
    update_interpreter_weights(&interpreter, theta, theta_len, weight_tensor_index);

    // 3️⃣ 模拟输入数据
    float input_data[10];
    for(int i=0;i<10;i++) input_data[i] = random(0,100)/100.0f;

    // 4️⃣ 推理
    float out = tflite_inference(input_data, 10, &interpreter);
    Serial.print("Inference output: "); Serial.println(out);

    // 5️⃣ 计算梯度 & EWC 微调
    float* grad = new float[theta_len];
    compute_dummy_gradient(grad, theta_len);
    float lr = 0.01f;
    float lambda_ewc = 0.5f;
    ewc_update(theta, theta_old, fisher_matrix, grad, theta_len, lr, lambda_ewc);

    // 6️⃣ 更新 Interpreter 权重
    update_interpreter_weights(&interpreter, theta, theta_len, weight_tensor_index);

    // 7️⃣ 保存到 SPIFFS
    save_float_bin("/model_weights.bin", theta, theta_len);

    // 清理
    delete[] theta;
    delete[] theta_old;
    delete[] fisher_matrix;
    delete[] grad;
    delete[] model_buffer;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    run_real_time_loop();
}

void loop() {
    delay(5000);
    run_real_time_loop();
}
