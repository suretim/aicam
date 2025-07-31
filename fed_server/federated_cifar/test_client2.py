import time
import tflite
import numpy as np
import paho.mqtt.client as mqtt
from sensor_module import Camera, HumiditySensor, LightSensor  # 假设有这些模块
from active_learning import uncertainty_sampling  # 不确定性主动学习
from federated_learning import update_local_model  # 联邦学习模型更新

# 初始化硬件模块
camera = Camera()
humidity_sensor = HumiditySensor()
light_sensor = LightSensor()

# 加载 TFLite 模型
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# MQTT 配置
mqtt_broker = "mqtt_broker_address"
mqtt_topic = "plant_health_prediction"
client = mqtt.Client()


# MQTT 连接函数
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # 客户端连接后订阅主题
    client.subscribe(mqtt_topic)


client.on_connect = on_connect
client.connect(mqtt_broker, 1883, 60)


# 图像采集和传感器数据采集函数
def get_data():
    # 获取图像数据
    image = camera.capture_image()  # 假设返回 64x64 图像
    # 获取传感器数据
    humidity = humidity_sensor.read_value()
    light = light_sensor.read_value()

    return image, humidity, light


# 不确定性主动学习函数
def perform_active_learning(image, humidity, light):
    # 在本地进行推理
    inputs = np.array([image, humidity, light])
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], inputs)
    interpreter.invoke()

    # 获取输出预测和不确定性评估
    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    uncertainty = uncertainty_sampling(prediction)  # 评估不确定性

    return prediction, uncertainty


# 训练本地模型函数（模拟）
def train_local_model(image, humidity, light):
    # 在本地进行模型更新（可以是增量训练或 finetuning）
    inputs = np.array([image, humidity, light])
    # 模拟更新模型（比如使用 TF Lite，增量训练）
    updated_model = update_local_model(inputs)
    return updated_model


# 上传模型更新到服务器
def upload_model_update(updated_model):
    # 模拟上传到服务器
    client.publish("federated_model_update", updated_model)
    print("Model update uploaded.")


# 主循环
def main_loop():
    while True:
        # 采集数据
        image, humidity, light = get_data()

        # 执行主动学习，获得预测和不确定性
        prediction, uncertainty = perform_active_learning(image, humidity, light)

        # 如果不确定性高，进行本地模型训练
        if uncertainty > threshold:  # 假设不确定性高于某个阈值
            updated_model = train_local_model(image, humidity, light)
            upload_model_update(updated_model)

        # 等待下一次采集
        time.sleep(10)  # 每 10 秒采集一次


if __name__ == "__main__":
    main_loop()
