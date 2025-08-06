#include "model_pb_handler.h"
#include "pb_decode.h"
#include <string.h>
#include <stdio.h>
// model_pb_handler.c
#if 1
/* 私有回调函数 - 只在当前文件使用 */
static bool decode_values_cb(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    float **values = (float **)*arg;
    size_t count = stream->bytes_left / sizeof(float);
    *values = malloc(count * sizeof(float));
    return pb_read(stream, *values, count * sizeof(float));
}

/* 公开的解码接口 */
bool decode_model_params(ModelParams *msg, uint8_t *data, size_t len) {
    float *value_buf = NULL;
    msg->values.funcs.decode = decode_values_cb;  // 绑定回调
    msg->values.arg = &value_buf;

    pb_istream_t stream = pb_istream_from_buffer(data, len);
    bool status = pb_decode(&stream, ModelParams_fields, msg);

    if (status && value_buf) {
        // 使用解码后的数据...
        free(value_buf);  // 释放内存
    }
    return status;
}
#else
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
#endif