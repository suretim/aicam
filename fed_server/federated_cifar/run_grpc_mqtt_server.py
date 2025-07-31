import grpc
import paho.mqtt.client as mqtt
import model_pb2
import model_pb2_grpc
from concurrent import futures
import time
import json

# MQTT配置
MQTT_BROKER = "192.168.0.57"
GRPC_SERVER = "192.168.133.128:50051"
MQTT_PORT = 1883
MQTT_TOPIC = "federated_model/parameters"
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#python -m venv .venv
#.\.venv\Scripts\activate  # Windows PowerShell
# 或 source .venv/bin/activate  # Linux/macOS

# 创建 MQTT 客户端
mqtt_client = mqtt.Client()
#mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# python -m grpc_tools.protoc --proto_path=./ --python_out=./ --grpc_python_out=./ model.proto
# start mqtt server D:\mqttserver\emqx-5.0.26-windows-amd64\bin\emqx.cmd
model_params = []
model_parameters_list = []
# 用于更新模型的函数
def publish_model_to_mqtt(model_parameters):
    """通过 MQTT 发布全局模型参数"""
    payload = json.dumps(model_parameters)  # 序列化为字符串
    mqtt_client.publish(MQTT_TOPIC, payload)
    print(f"Published model parameters to MQTT: {payload}")



class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.model_parameters_list = []
    def UploadModelxParams0(self, request, context):
        # 模拟处理模型参数
        print(f"Received model params from client {request.client_id}")
        print(f"weights: {request.weights[:5]} ...")  # 只打印前5个 float 防止太长

        return model_pb2.ServerResponse(
            message="Model parameters successfully updated.",
            update_successful=True,
            update_timestamp=int(time.time())
        )

    def GetUpdateStatus(self, request, context):
        # 假设总是成功并返回状态
        return model_pb2.ServerResponse(
            message="Model update status fetched successfully.",
            update_successful=True,
            update_timestamp=int(time.time())
        )

    # 假设你希望每次收到一个 client 模型参数都加入一个列表后聚合


    def federated_avg(self,model_parameters_list):
        """
        简单的 FedAvg 实现：对多个客户端上传的模型参数（float 数组）取平均
        参数:
            model_parameters_list: List of List[float]
        返回:
            List[float]: 平均后的模型参数
        """
        if not model_parameters_list:
            raise ValueError("model_parameters_list is empty")

        num_clients = len(model_parameters_list)
        num_params = len(model_parameters_list[0])

        # 初始化为 0
        avg_params = [0.0] * num_params

        for params in model_parameters_list:
            for i in range(num_params):
                avg_params[i] += params[i]

        # 求平均
        avg_params = [x / num_clients for x in avg_params]

        return avg_params

    def UploadModelParams(self, request, context):
        """
        更新全局模型并通过 MQTT 发布
        """
        try:
            client_params = list(request.weights)  # 需要转换为 list
            print("Received model parameters: ", client_params)

            self.model_parameters_list.append(client_params)

            # 聚合
            new_model_parameters = self.federated_avg(self.model_parameters_list)

            # 发布新模型参数
            publish_model_to_mqtt(new_model_parameters)
            # 返回响应
            # return model_pb2.UpdateResponse(status="Success")
            success = True  # Let's assume the update is successful for this example
            timestamp = int(time.time())  # Get current timestamp

            # Return response with a success message and timestamp
            return model_pb2.ServerResponse(
                message="Model parameters successfully updated.",
                update_successful=success,
                update_timestamp=timestamp)

        except Exception as e:
            print("Error during UploadModelParams:", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            success = False  # Let's assume the update is successful for this example
            timestamp = int(time.time())  # Get current timestamp
            return model_pb2.ServerResponse(
                message="Model parameters none successfully updated.",
                update_successful=success,
                update_timestamp=timestamp)




# MQTT 客户端回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully!")
        # 连接成功后，订阅一个主题
        client.subscribe("model/update")
    else:
        print("Failed to connect, return code:", rc)


def on_message(client, userdata, msg):
    print(f"Received MQTT message: {msg.payload.decode()}")

    try:
        # 尝试解析 JSON 并提取参数
        message = json.loads(msg.payload.decode())
        weights = message.get('weights')

        if not isinstance(weights, list):
            raise ValueError("Invalid format: 'weights' must be a list")

        print(f"Updated model parameters: {weights}")

        # 建立 gRPC 通信
        grpc_channel = grpc.insecure_channel(GRPC_SERVER)
        stub = model_pb2_grpc.FederatedLearningStub(grpc_channel)

        # 构建 gRPC 请求
        request = model_pb2.ModelParams(client_id=1, weights=weights)

        # 调用远程接口
        response = stub.UploadModelParams(request)
        print(f"gRPC server response: {response.message}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode MQTT message as JSON: {e}")
    except grpc.RpcError as e:
        print(f"gRPC communication failed: {e.details()} (code: {e.code()})")
    except Exception as e:
        print(f"Unexpected error in on_message: {e}")



def mqtt_subscribe():
    #client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    # 设置用户名和密码
    username = "tim"  # 替换为你的 MQTT 用户名
    password = "tim"  # 替换为你的 MQTT 密码
    mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
    # 设置重连超时时间，单位为毫秒
    reconnect_timeout_ms = 10000  # 10秒的重连超时
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

    #client.connect("192.168.0.57", 1883, 60)  # 更换为实际的 MQTT broker 地址
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    # 让程序持续运行，以便接收和处理消息
    try:
        while True:
            time.sleep(1)  # 可以适当调整为更小的时间间隔，确保不中断 MQTT 事件处理
    except KeyboardInterrupt:
        print("Disconnected from MQTT broker.")
        mqtt_client.loop_stop()

def serve ():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()
    server.wait_for_termination()

def serve1 ():
    # 启动 gRPC 服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)

    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()
    #server.wait_for_termination()
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
