#pragma once

  esp_err_t init_esp32_camera(void);
 esp_err_t GetESPImage(int image_width, int image_height, int channels,   float* image_data);