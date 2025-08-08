import grpc
import paho.mqtt.client as mqtt
import model_pb2
import model_pb2_grpc
from concurrent import futures
import time
import datetime
import json
import threading
import numpy as np

# MQTT配置
#MQTT_BROKER = "192.168.0.57"
MQTT_BROKER = "127.0.0.1"
#GRPC_SERVER = "192.168.133.128:50051"
GRPC_SERVER = "127.0.0.1:50051"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
MSG_PUBLISH = "msg/mqttx_"  # 替换为你的主题
#define MQTT_TOPIC_SUB "capture/mqttx_"

GRPC_SUBSCRIBE = "grpc_sub/weights"

#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#python -m venv .venv
#.\.venv\Scripts\activate  # Windows PowerShell
# 或 source .venv/bin/activate  # Linux/macOS


# python -m grpc_tools.protoc --proto_path=./ --python_out=./ --grpc_python_out=./ model.proto
# start mqtt server D:\mqttserver\emqx-5.0.26-windows-amd64\bin\emqx.cmd
model_params = []
model_parameters_list = []
new_model_parameters=[]


def publish_message():
    """每分钟发布消息的定时任务"""
    while True:
        # 生成带时间戳的消息
        message = f"定时消息 @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        #message = f"weight/mqtrx_"

        # 发布消息
        result = mqtt_client.publish(MSG_PUBLISH, message, qos=1)

        # 检查发布状态
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"已发布: {message} → [{MQTT_PUBLISH}]")
        else:
            print(f"发布失败，错误码: {result.rc}")

        # 等待60秒
        time.sleep(180)


# 用于更新模型的函数
def publish_model_to_mqtt(model_parameters):
    # 如果是 numpy 数组，先转成列表
    if isinstance(model_parameters, np.ndarray):
        model_parameters = model_parameters.tolist()
    elif isinstance(model_parameters, list) and isinstance(model_parameters[0], np.ndarray):
        model_parameters = [w.tolist() for w in model_parameters]
    # 构建消息
    msg_weights = model_pb2.ModelParams()
    msg_weights.param_type = model_pb2.CLASSIFIER_WEIGHT
    msg_weights.values.extend(model_parameters.flatten().tolist())
    msg_weights.client_id = 1  # 可选设置 client_id
    payload_weights = msg_weights.SerializeToString()
    msg_bias = model_pb2.ModelParams()
    msg_bias.param_type = model_pb2.CLASSIFIER_BIAS
    msg_bias.values.extend(model_parameters.flatten().tolist())
    msg_bias.client_id = 2  # 可选设置 client_id
    payload_bias = msg_bias.SerializeToString()
    # 打包为 JSON 格式
    #weights_data = {
    #    "mqtrx_weights": model_parameters,
        # "metadata": {
        #     "num_classes": 5,
        #     "input_shape": 64
        # }
    #}
    """通过 MQTT 发布全局模型参数"""
    #payload = json.dumps(weights_data)  # 序列化为字符串


    mqtt_client.publish(FEDER_PUBLISH, payload_weights)
    mqtt_client.publish(FEDER_PUBLISH, payload_bias)
    print(f"Published model parameters to MQTT: {payload_bias}")

from server_h5 import ESP32TOH5 as store_h5

class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.model_parameters_list = []
        self.model_labels_list = []
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
            client_params = list(request.values)  # 需要转换为 list
            print("Received model parameters: ", client_params)
            # 使用示例
            #params_array =np.random.rand(100, 64).astype(np.float32)  # 模擬ESP32輸出 client_params[1:64]  #
            #labels_array = np.random.randint(0, 3, size=100) # 模擬ESP32輸出 client_params[0]  #
            params_array = np.array(client_params[1:], dtype=np.float32)  # Convert to NumPy array
            labels_array = np.array([ client_params[0]], dtype=np.float32)  # Convert to NumPy array

            if params_array.shape[1] != self.model_parameters_list.shape[1]:
                self.model_parameters_list = np.empty((0, 64))
                self.model_labels_list = np.empty((0,))
                #raise ValueError(
                #    f"Expected {self.model_parameters_list.shape[1]} features, got {params_array.shape[1]}")
            else:
                # 使用np.vstack进行垂直堆叠
                self.model_parameters_list = np.vstack((self.model_parameters_list, params_array))
                self.model_labels_list = np.concatenate((self.model_labels_list, labels_array))


            # 聚合
            #self.model_parameters_list.append(params_array)
            #self.model_labels_list.append(labels_array)

            if self.model_parameters_list.shape[0]==2:
                parameters_avg = self.federated_avg(self.model_parameters_list)
                arravg = np.array(parameters_avg)
                print("federated_avg parameters: ", arravg)
                #features = np.round(features, decimals=3)  # Round to 1 decimal
                #print("federated features: ", features)

                data_dir = "data"
                device_id = "client_002"
                data_gen = store_h5(data_dir,device_id)

                data_gen.save_esp32_features(
                    features=params_array,
                    labels=labels_array
                    #metadata={}
                )

            # 发布新模型参数
            #publish_model_to_mqtt(new_model_parameters)
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
        client.subscribe(GRPC_SUBSCRIBE)
    else:
        print("Failed to connect, return code:", rc)


def on_message(client, userdata, msg):
    print(f"Received MQTT message: {msg.payload.decode()}")

    try:
        # 尝试解析 JSON 并提取参数
        message = json.loads(msg.payload.decode())
        fea_weights = message.get('fea_weights')
        fea_labels = message.get('fea_label')


        # 提取特征权重和标签
        #fea_weights = message['fea_weights']  # 64维特征向量
        #fea_labels = message['fea_label'][0]  # 单个标签值(1)
        fea_vec = fea_weights.extend(fea_labels)
        fea_vec= fea_labels+fea_weights
        print(f"Updated model parameters: {fea_vec}")

        if not isinstance(fea_vec, list):
            raise ValueError("Invalid format: 'fea_vec' must be a list")


        # 建立 gRPC 通信
        grpc_channel = grpc.insecure_channel(GRPC_SERVER)
        stub = model_pb2_grpc.FederatedLearningStub(grpc_channel)

        # 构建 gRPC 请求
        request = model_pb2.ModelParams(client_id=1, values= fea_vec)

        # 调用远程接口
        response = stub.UploadModelParams(request)
        print(f"gRPC server response: {response.message}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode MQTT message as JSON: {e}")
    except grpc.RpcError as e:
        print(f"gRPC communication failed: {e.details()} (code: {e.code()})")
    except Exception as e:
        print(f"Unexpected error in on_message: {e}")


# 创建 MQTT 客户端
mqtt_client = mqtt.Client()
# client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
# 设置用户名和密码
username = "tim"  # 替换为你的 MQTT 用户名
password = "tim"  # 替换为你的 MQTT 密码
mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
# 设置重连超时时间，单位为毫秒
reconnect_timeout_ms = 10000  # 10秒的重连超时
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)


def mqtt_subscribe():
    mqtt_client.loop_start()
    # 让程序持续运行，以便接收和处理消息
    try:
        while True:
            time.sleep(1)  # 可以适当调整为更小的时间间隔，确保不中断 MQTT 事件处理
    except KeyboardInterrupt:
        print("Disconnected from MQTT broker.")
        mqtt_client.loop_stop()

def serve0 ():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()
    server.wait_for_termination()

def serve():
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

    try:
        #from threading import Thread
        #thread = Thread(target=mqtt_subscribe)
        subcribe_thread = threading.Thread(target=mqtt_subscribe)
        subcribe_thread.start()
        '''
        # 创建定时发布线程
        publish_thread = threading.Thread(target=publish_message)
        publish_thread.daemon = True  # 设为守护线程
        publish_thread.start()
        '''
        serve()

    except KeyboardInterrupt:
        print("\n程序终止")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        mqtt_client.disconnect()
        mqtt_client.loop_stop()
        print("MQTT连接已关闭")