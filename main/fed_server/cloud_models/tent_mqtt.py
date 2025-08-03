import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# ==== 推理后的结果示例 ====
tent_id = "TENT_A01"
predicted_class = "disease"  # or "healthy"
distance = 0.22  # 可选，用于置信度参考

# ==== MQTT 参数配置 ====
MQTT_BROKER = "192.168.68.237"      # 改为你自己的 broker 地址
MQTT_PORT = 1883
MQTT_TOPIC = "smartagriculture/leaf_detection"

# 创建 MQTT 客户端并连接
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# 构造上传的 JSON 消息
payload = {
    "tent_id": tent_id,
    "timestamp": datetime.now().isoformat(),
    "prediction": predicted_class,
    "distance": round(distance, 3)
}

# 发送消息
client.publish(MQTT_TOPIC, json.dumps(payload))
print(f"📡 Sent MQTT message to [{MQTT_TOPIC}]:\n{json.dumps(payload, indent=2)}")

client.disconnect()
