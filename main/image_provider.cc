
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"

//#include "app_camera_esp.h"
#include "esp_camera.h"
//#include "model_settings.h"
//#include "image_provider.h"
//#include "esp_main.h" 

constexpr int kNumCols = 64;
constexpr int kNumRows = 64;
constexpr int kNumChannels = 1;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 2;
constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
extern const char* kCategoryLabels[kCategoryCount];  


static const char* TAG = "app_camera";

static uint16_t *display_buf; // buffer to hold data to be sent to display
//#define DISPLAY_SUPPORT 1
// Get the camera module ready 
// void *image_provider_get_display_buf()
// {
//   return (void *) display_buf;
// }
 
