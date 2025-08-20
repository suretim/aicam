#include <WiFi.h>
#include <PubSubClient.h>
#include "FS.h"
#include "SPIFFS.h"
#include "Base64.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <map>

#define TAG "MQTT"
#if 0
    //#define MQTT_BROKER_URI "mqtt://192.168.133.129:1883"
    #define MQTT_BROKER_URI "mqtt://192.168.68.237:1883"
#else
    //#define MQTT_BROKER_URI "mqtt://192.168.0.57:1883"
    #define MQTT_BROKER_URI "mqtt://127.0.0.1:1883"
#endif
#define MQTT_USERNAME "tim"
#define MQTT_PASSWORD "tim"
#define MQTT_CLIENT_ID_PREFIX "mqttx_" 
#define MQTT_TOPIC_PUB "grpc_sub/weights"
//GRPC_SUBSCRIBE = "grpc_sub/weights"

#define MQTT_TOPIC_SUB "federated_model/parameters"
#define MQTT_KEEPALIVE_SECONDS 120


const char* ssid = "1573";
const char* password = "12345678";
const char* mqtt_server = "127.0.0.1";
WiFiClient espClient;
PubSubClient client(espClient);

struct FileState {
  File file;
  int last_chunk = -1;
  bool done = false;
};

std::map<String, FileState> file_states;
std::map<String, String> topics = {
  {"plant/ewc/model_weights", "/model_weights.tflite"},  // 假設已轉 tflite
  {"plant/ewc/fisher_matrix", "/fisher_matrix.npz"}
};

void run_inference(){
  Serial.println("Starting inference...");

  // 1. 讀取 tflite 模型
  File model_file = SPIFFS.open("/model_weights.tflite", FILE_READ);
  if(!model_file){
    Serial.println("Model not found");
    return;
  }

  size_t model_size = model_file.size();
  uint8_t* model_buffer = new uint8_t[model_size];
  model_file.read(model_buffer, model_size);
  model_file.close();

  const tflite::Model* model = tflite::GetModel(model_buffer);
  if(model->version() != TFLITE_SCHEMA_VERSION){
    Serial.println("Model schema mismatch");
    delete[] model_buffer;
    return;
  }

  // 2. 建立 interpreter
  constexpr int tensor_arena_size = 10 * 1024;
  static uint8_t tensor_arena[tensor_arena_size];
  tflite::MicroInterpreter interpreter(model, nullptr, tensor_arena, tensor_arena_size, nullptr);
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if(allocate_status != kTfLiteOk){
    Serial.println("Failed to allocate tensors");
    delete[] model_buffer;
    return;
  }

  // 3. 準備輸入資料 (此處為示例，可改成 sensor 資料)
  TfLiteTensor* input = interpreter.input(0);
  for(int i=0;i<input->bytes;i++){
    input->data.f[i] = 0.0f; // 模擬輸入
  }

  // 4. 執行推理
  if(interpreter.Invoke() != kTfLiteOk){
    Serial.println("Inference failed");
  } else {
    Serial.println("Inference success");
    // 輸出示例
    TfLiteTensor* output = interpreter.output(0);
    Serial.printf("Output[0] = %f\n", output->data.f[0]);
  }

  delete[] model_buffer;
}

void saveChunkToFile(int chunk_id, const String &b64, File &file){
  int decoded_len;
  uint8_t* decoded = base64_decode(b64.c_str(), b64.length(), &decoded_len);
  file.write(decoded, decoded_len);
  free(decoded);
}

void callback(char* topic, byte* payload, unsigned int length){
  String msg = "";
  for(unsigned int i=0;i<length;i++){
    msg += (char)payload[i];
  }

  // 解析 JSON
  DynamicJsonDocument doc(1024);
  DeserializationError err = deserializeJson(doc, msg);
  if(err){
    Serial.println("JSON parse error");
    return;
  }

  int chunk_id = doc["chunk_id"];
  String data = doc["data"];

  String path = topics[String(topic)];
  FileState &state = file_states[path];

  if(chunk_id == -1){
    if(state.file){
      state.file.close();
      state.done = true;
      Serial.printf("File %s received completely\n", path.c_str());
    }

    // 如果所有檔案都完成，觸發推理
    bool all_done = true;
    for(auto &kv:file_states){
      if(!kv.second.done){
        all_done = false;
        break;
      }
    }
    if(all_done){
      run_inference();
    }
    return;
  }

  if(chunk_id <= state.last_chunk){
    return; // 已接收過
  }

  if(!state.file){
    state.file = SPIFFS.open(path, FILE_WRITE);
    if(!state.file){
      Serial.printf("Failed to open %s\n", path.c_str());
      return;
    }
  }

  saveChunkToFile(chunk_id, data, state.file);
  state.last_chunk = chunk_id;
  Serial.printf("Received chunk %d for %s\n", chunk_id, path.c_str());
}

void setup() {
  Serial.begin(115200);
  SPIFFS.begin(true);

  for(auto &kv:topics){
    file_states[kv.second] = FileState();
  }

  WiFi.begin(ssid, password);
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");

  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  while(!client.connected()){
    if(client.connect("ESP32_EWC_Receiver")){
      for(auto &kv: topics){
        client.subscribe(kv.first.c_str());
      }
      Serial.println("MQTT connected and subscribed");
    } else {
      delay(500);
    }
  }
}

void loop() {
  client.loop();
}
