import grpc
import paho.mqtt.client as mqtt
import model_pb2
import model_pb2_grpc
from concurrent import futures
import time
import json

# 全局变量用于存储模型参数
# start mqtt server D:\mqttserver\emqx-5.0.26-windows-amd64\bin\emqx.cmd
model_params = []


# gRPC 服务定义
class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def UploadModelParams(self, request, context):
        print(f"Received model parameters via gRPC from client {request.client_id}")
        print(f"Weights: {request.weights}")
        return model_pb2.ServerResponse(message="Model parameters received via gRPC")


# MQTT 客户端回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully!")
        # 连接成功后，订阅一个主题
        client.subscribe("model/params")
    else:
        print("Failed to connect, return code:", rc)

def on_message (client, userdata, msg):
    global model_params
    print(f"Received MQTT message: {msg.payload.decode()}")
    # 将 MQTT 消息解析成模型参数
    try:
        message = json.loads(msg.payload.decode())
        model_params = message['weights']  # 假设传递的消息格式为 {'weights': [...]}

        print(f"Updated model parameters: {model_params}")

        # 将接收到的参数通过 gRPC 上传到服务器
        grpc_channel = grpc.insecure_channel('192.168.0.57:50051')
        stub = model_pb2_grpc.FederatedLearningStub(grpc_channel)

        request = model_pb2.ModelParams(client_id=1, weights=model_params)
        response = stub.UploadModelParams(request)
        print(f"gRPC server response: {response.message}")

    except Exception as e:
        print(f"Error processing MQTT message: {e}")


def mqtt_subscribe():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    # 设置用户名和密码
    username = "tim"  # 替换为你的 MQTT 用户名
    password = "tim"  # 替换为你的 MQTT 密码
    client.username_pw_set(username, password)  # 设置用户名和密码
    # 设置重连超时时间，单位为毫秒
    reconnect_timeout_ms = 10000  # 10秒的重连超时
    client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

    client.connect("192.168.0.57", 1883, 60)  # 更换为实际的 MQTT broker 地址
    client.loop_start()

    # 让程序持续运行，以便接收和处理消息
    try:
        while True:
            time.sleep(1)  # 可以适当调整为更小的时间间隔，确保不中断 MQTT 事件处理
    except KeyboardInterrupt:
        print("Disconnected from MQTT broker.")
        client.loop_stop()

def serve():
    # 启动 gRPC 服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)

    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()

    try:
        while True:
            time.sleep(60 * 60 * 24)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    # 启动 gRPC 服务器和 MQTT 客户端
    from threading import Thread

    thread = Thread(target=mqtt_subscribe)
    thread.start()

    serve()
