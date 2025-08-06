#include "model_pb_handler.h"
#include "pb_decode.h"
#include <string.h>
#include <stdio.h>

bool decode_model_params(const uint8_t *payload, size_t length, ParsedModelParams *out_params) {
    ModelParams msg = ModelParams_init_zero;

    pb_istream_t stream = pb_istream_from_buffer(payload, length);

    // 临时数组接收 repeated float
    float value_buf[MAX_PARAM_LENGTH] = {0};
    msg.values.values = value_buf;
    msg.values.size = MAX_PARAM_LENGTH;

    if (!pb_decode(&stream, ModelParams_fields, &msg)) {
        printf("Decoding failed: %s\n", PB_GET_ERROR(&stream));
        return false;
    }

    out_params->param_type = msg.param_type;
    out_params->value_count = msg.values.size;
    memcpy(out_params->values, msg.values.values, sizeof(float) * msg.values.size);
    out_params->client_id = msg.client_id;

    return true;
}
