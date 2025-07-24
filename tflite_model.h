#pragma once
namespace tflite {
  class Model {
  public:
    int version_ = 3;
    int version() const { return version_; }
  };
  const Model* GetModel(const unsigned char* buf);
} 