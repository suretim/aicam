#pragma once

  esp_err_t init_esp32_camera(void);
 esp_err_t GetImage(int image_width, int image_height, int channels, uint8_t* image_data);