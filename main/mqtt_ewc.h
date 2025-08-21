// mqtt_ewc.h
#pragma once
#include <vector>
#include <stdint.h>

void mqtt_init();
bool ewc_assets_received();
void parse_ewc_assets();

// trainable + fisher
extern std::vector<std::vector<float>> trainable_layers;
extern std::vector<std::vector<float>> fisher_layers;
extern std::vector<std::vector<int>> layer_shapes;
