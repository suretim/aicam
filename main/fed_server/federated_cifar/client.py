import numpy as np
import paho.mqtt.client as mqtt
import model_pb2
import json

# 模拟你训练得到的权重
# 构造假数据
dense_weights = np.random.rand(64, 3).astype(np.float32)

# 构建消息
msg = model_pb2.ModelParams()
msg.param_type = model_pb2.CLASSIFIER_WEIGHT
msg.values.extend(dense_weights.flatten().tolist())
msg.client_id = 1  # 可选设置 client_id

# 序列化消息
payload = msg.SerializeToString()
#json_payload = json.dumps(payload)

# 发布到 MQTT

MQTT_BROKER = "192.168.0.57"
#MQTT_BROKER = "192.168.68.237"
#MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
mqtt_client = mqtt.Client()

def mqtt_pub_metadata():
    #client = mqtt.Client()
    #mqtt_client.on_connect = on_connect
    #mqtt_client.on_message = on_message
    # 设置用户名和密码
    username = "tim"  # 替换为你的 MQTT 用户名
    password = "tim"  # 替换为你的 MQTT 密码
    mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
    # 设置重连超时时间，单位为毫秒
    reconnect_timeout_ms = 10000  # 10秒的重连超时
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)  # 更换为实际的 MQTT broker 地址

    mqtt_client.publish(FEDER_PUBLISH, payload)
    mqtt_client.disconnect()


mqtt_pub_metadata()

