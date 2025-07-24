#include "tflite_model.h"
namespace tflite {
  const Model* GetModel(const unsigned char* buf) {
    return reinterpret_cast<const Model*>(buf);
  }
}