#include "tflite_model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// namespace tflite {
//   const Model* GetModel(const unsigned char* buf) {
//     return reinterpret_cast<const Model*>(buf);
//   }
// }


// Globals, used for compatibility with Arduino-style sketches.
namespace ftlite_fed{
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
//FeatureProvider* feature_provider = nullptr;
//RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[64];
int8_t* model_input_buffer = nullptr;
}  // namespace

