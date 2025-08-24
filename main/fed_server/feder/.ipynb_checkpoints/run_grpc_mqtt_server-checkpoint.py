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
from utils import DataLoader

from utils import DataSaver
from utils import LeamPipeline
import tensorflow as tf
import os

# MQTT配置
#MQTT_BROKER = "192.168.0.57"
MQTT_BROKER = "127.0.0.1"
GRPC_SERVER = "127.0.0.1:50051"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
WEIGHT_FISH_PUBLISH = "ewc/weight_fisher"
FISH_SHAP_PUBLISH = "ewc/layer_shapes"

GRPC_SUBSCRIBE = "grpc_sub/weights"

EWC_ASSETS="ewc_assets"

#define MQTT_TOPIC_PUB "grpc_sub/weights"
#define MQTT_TOPIC_SUB "federated_model/parameters"
#define WEIGHT_FISH_SUB "ewc/weight_fisher"
#define FISH_SHAP_SUB  "ewc/layer_shapes"

client_request_code= 1
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#python -m venv .venv
#.\.venv\Scripts\activate  # Windows PowerShell
# 或 source .venv/bin/activate  # Linux/macOS
# python -m grpc_tools.protoc --proto_path=./ --python_out=./ --grpc_python_out=./ model.proto
# start mqtt server D:\mqttserver\emqx-5.0.26-windows-amd64\bin\emqx.cmd
#model_params = []
#model_parameters_list = []
#new_model_parameters=[]

class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self,data_dir=None,mqtt_client=None):
        #self.model_parameters_list = []
        #self.model_labels_list = []
        self.data_dir = data_dir
        self.mqtt_client = mqtt_client
        self.model_parameters_list = np.empty((0, 64))
        self.model_labels_list = np.empty((0,))
        self.client_id=None

    # 用于更新模型的函数
    @classmethod
    def publish_model_to_mqtt(cls,model_parms1,model_parms2,client_id):
        # 如果是 numpy 数组，先转成列表
        par1 =model_parms1
        par2 =model_parms2
        if isinstance(par1, np.ndarray):
            par1 = par1.tolist()
        elif isinstance(par1, list) and isinstance(par1[0], np.ndarray):
            par1 = [w.tolist() for w in par1]
        if isinstance(par2, np.ndarray):
            par2 = par2.tolist()
        elif isinstance(par2, list) and isinstance(par2[0], np.ndarray):
            par2 = [w.tolist() for w in par2]

        # 构建消息
        msg_weights = model_pb2.ModelParams()
        msg_weights.param_type = model_pb2.CLASSIFIER_WEIGHT
        msg_weights.values.extend(par1.flatten().tolist())
        msg_weights.client_id = client_id  # 可选设置 client_id
        payload_weights = msg_weights.SerializeToString()
        mqtt_client.publish(FEDER_PUBLISH, payload_weights)
        print(f"Published model parameters to MQTT: {payload_weights}")
        msg_bias = model_pb2.ModelParams()
        msg_bias.param_type = model_pb2.CLASSIFIER_BIAS
        msg_bias.values.extend(par2.flatten().tolist())
        msg_bias.client_id = client_id  # 可选设置 client_id
        payload_bias = msg_bias.SerializeToString()
        mqtt_client.publish(FEDER_PUBLISH, payload_bias)
        print(f"Published model parameters to MQTT: {payload_bias}")
        # 打包为 JSON 格式
        # weights_data = {
        #    "mqtrx_weights": model_parameters,
        # "metadata": {
        #     "num_classes": 5,
        #     "input_shape": 64
        # }
        # }
        """通过 MQTT 发布全局模型参数"""
        # payload = json.dumps(weights_data)  # 序列化为字符串
 

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
        data_loader = DataLoader(data_dir=data_dir, device_id="client_001")
        pipeline = LeamPipeline(data_loader =data_loader)
        devices = pipeline.get_available_devices()
        federated_data = pipeline.get_federated_dataset(devices=devices, samples_per_device=500)
        target_device =self.client_id
        pipeline.load_available_devices(target_device)
        pipeline.load_available_devices()
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
        # 发布新模型参数

        self.publish_model_to_mqtt(avg_params)



    def UploadModelParams(self, request, context):
        """
        更新全局模型并通过 MQTT 发布
        """
        client_id=request.client_id
        print(f"收到来自客户端 {request.client_id} 的参数")
        try:
            client_params = list(list(request.values) ) # 需要转换为 list
            #print("Received model parameters: ", client_params)
            #print("request.client_id  :",client_id)
            #print("client_params 結構:", client_params)
            #print("第一行類型:", type(client_params[0]))
            GROUP_SIZE = 65
            num_groups = len(client_params) // GROUP_SIZE

            # 转换为 NumPy 数组并重新组织
            data = np.array(client_params, dtype=np.float32).reshape(num_groups, GROUP_SIZE)
            labels_array = data[:, 0].astype(np.int32)  # 所有行的第 0 列（标签）
            params_array = data[:, 1:65]  # 所有行的第 1 列之后（特征）
            # 使用示例
            #params_array =np.random.rand(100, 64).astype(np.float32)  # 模擬ESP32輸出 client_params[1:64]  #
            #labels_array = np.random.randint(0, 3, size=100) # 模擬ESP32輸出 client_params[0]  #
            #params_array = np.array(client_params[1:], dtype=np.float32)  # Convert to NumPy array
            #labels_array = np.array([ client_params[0]], dtype=np.float32)  # Convert to NumPy array

            #labels_array = np.array([x[0] for x in client_params], dtype=np.int32)
            # 提取所有行的第 1 列之后（特征）
            #params_array = np.array([x[1:] for x in client_params], dtype=np.float32)
            #print("Received labels_array: ", labels_array )
            #print("Received params_array: ", params_array )


            # 初始化存储列表（如果是第一次运行）
            if not hasattr(self, 'model_parameters_list'):
                self.model_parameters_list = np.empty((0, 64))  # 特征维度 64
                self.model_labels_list = np.empty((0,))  # 标签

            # 检查维度一致性
            if params_array.shape[1] != self.model_parameters_list.shape[1]:
                print(f"维度不匹配！重置存储列表。",params_array.shape[1] )
                self.model_parameters_list = np.empty((0, 64))
                self.model_labels_list = np.empty((0,))

            # 追加数据
            self.model_parameters_list = np.vstack((self.model_parameters_list, params_array))
            self.model_labels_list = np.concatenate((self.model_labels_list, labels_array))
            # 聚合
            #self.model_parameters_list.append(params_array)
            #self.model_labels_list.append(labels_array)
            print("Received model_parameters_list: ", self.model_parameters_list.shape[0])
            print("Received model_labels_list: ", self.model_labels_list.shape[0])

            if self.model_parameters_list.shape[0]>=10:
                #parameters_avg = self.federated_avg(self.model_parameters_list)
                #arravg = np.array(parameters_avg)
                #print("federated_avg parameters: ", arravg)
                #features = np.round(features, decimals=3)  # Round to 1 decimal
                #print("federated features: ", features)

                #data_dir = "../../../../data"
                #device_id = "client_003"
                data_gen = DataSaver(data_dir,client_id)

                data_gen.save_features(
                    features=self.model_parameters_list,
                    labels=self.model_labels_list
                    #metadata={}
                )
                self.model_parameters_list = np.empty((0, 64))
                self.model_labels_list = np.empty((0,))
                print("Model parameters successfully updated." )

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
                message="Model parameters none successfully updated."+client_id,
                update_successful=success,
                update_timestamp=timestamp)



#end of FederatedLearningServicer



def save_fisher_matrix_to_bin(fisher_matrix, bin_file_path):
    # Open the binary file in write mode
    with open(bin_file_path, 'wb') as bin_file:
        for matrix in fisher_matrix:
            # Convert each matrix (numpy array) to raw bytes
            matrix_bytes = matrix.numpy().tobytes()
            bin_file.write(matrix_bytes)  # Write the bytes to the file
    print(f"Fisher matrix saved to {bin_file_path}")
 
    
def pubish_fisher_matrix(client, topic, bin_file_path):
    with open(bin_file_path, 'rb') as f:
        payload = f.read()  # Read the binary content of the .bin file
        client.publish(topic, payload)  # Send the binary data as the MQTT message
        print(f"Fisher matrix sent to topic {topic}")

def save_ewc_assets_to_bin(model, save_dir="../lstm/ewc_assets"):
    # Load model weights
    #model.load_weights(os.path.join(save_dir, "model_weights.h5"))
    
    # Load Fisher matrix
    fisher_data = np.load(os.path.join(save_dir, "fisher_matrix.npz"))
    fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]
    
    print(f"EWC assets loaded from {save_dir}")
    save_fisher_matrix_to_bin(fisher_matrix,os.path.join(save_dir, "fisher_matrix.bin"))
     
    return fisher_matrix

  

    
def load_ewc_assets(save_dir="../lstm/ewc_assets"):
    fisher_data = np.load(f"{save_dir}/fisher_matrix.npz")
    fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]
    return fisher_matrix


def publish_message():
    global client_request_code
    while True:
        if client_request_code >=0 :
            fisher_matrix = load_ewc_assets()
            message = b''.join([arr.numpy().tobytes() for arr in fisher_matrix])
            #message=load_ewc_assets(model, save_dir="../lstm/ewc_assets")
            #message = f"定时消息 @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            # 发布消息
            client_request_code=client_request_code+1
            result = mqtt_client.publish(WEIGHT_FISH_PUBLISH, message, qos=1)
            # 检查发布状态
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                #print(f"已发布: {message} → [{WEIGHT_FISH_PUBLISH}]")
                print(f"已发布:   [{WEIGHT_FISH_PUBLISH}]",client_request_code)
            else:
                print(f"发布失败，错误码: {result.rc}")
        # 等待180秒
        time.sleep(30)


# MQTT 客户端回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully!")
        # 连接成功后，订阅一个主题
        client.subscribe(GRPC_SUBSCRIBE)
    else:
        print("Failed to connect, return code:", rc)


def on_message(client, userdata, msg):
    global client_request_code
    try:
        # 尝试解析 JSON 并提取参数
        #message = parse_message(msg.payload.decode())
        message = json.loads(msg.payload.decode())
        client_request  = message.get('client_request','0')
        client_id = message.get('client_id', '1')
        print(f"Received MQTT message: {msg.payload.decode()}",client_request ,client_request_code,client_id)

        if client_request ==1 :
            #publish_message()
            client_request_code = 0
        else:
            #client_request_code = 1
            fea_weights = message.get('fea_weights', [])
            fea_labels = message.get('fea_labels', [])  # <-- 改成复数

            # 确保都是 list
            if not isinstance(fea_weights, list):
                fea_weights = [fea_weights]
            if not isinstance(fea_labels, list):
                fea_labels = [fea_labels]

            # 拼接 label + features
            fea_vec = fea_labels + fea_weights
            print(f"Updated model parameters: {fea_vec}")

            #fea_weights = message.get('fea_weights' )
            #fea_labels = message.get('fea_label' )  # Get first element or None

            #fea_vec = message.get('fea_label', []) + message.get('fea_weights', [])
            # 提取特征权重和标签
            #fea_weights = message['fea_weights']  # 64维特征向量
            #fea_labels = message['fea_label'][0]  # 单个标签值(1)
            #fea_vec = fea_labels .extend(fea_weights)
            #

            if not isinstance(fea_vec, list):
                raise ValueError("Invalid format: 'fea_vec' must be a list")
            #load_ewc_assets(model, save_dir=EWC_ASSETS)
            #pubish_fisher_matrix(client=client, topic=MSG_PUBLISH, bin_file_path=os.path.join(EWC_ASSETS, "fisher_matrix.bin"))

            # 建立 gRPC 通信
            grpc_channel = grpc.insecure_channel(GRPC_SERVER)
            stub = model_pb2_grpc.FederatedLearningStub(grpc_channel)

            # 构建 gRPC 请求
            request = model_pb2.ModelParams(client_id=client_id, values= fea_vec)

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
    mqtt_client.loop_start()
    # 让程序持续运行，以便接收和处理消息
    try:
        while True:
            time.sleep(1)  # 可以适当调整为更小的时间间隔，确保不中断 MQTT 事件处理
    except KeyboardInterrupt:
        print("Disconnected from MQTT broker.")
        mqtt_client.loop_stop()

def serve(data_dir,mqtt_client):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(data_dir=data_dir,mqtt_client=mqtt_client), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()
    server.wait_for_termination()



#conda activate my_env
#cd C:\tim\aicam\main\fed_server\cloud_models
#python emqx_manager.py
#netstat -ano | findstr :18083

if __name__ == '__main__':
# 启动 gRPC 服务器和 MQTT 客户端

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
    data_dir="../../../../data"
    try:
        #
        #from threading import Thread
        #thread = Thread(target=mqtt_subscribe)
        subcribe_thread = threading.Thread(target=mqtt_subscribe)
        subcribe_thread.start()
         
        # 创建定时发布线程
        publish_thread = threading.Thread(target=publish_message)
        publish_thread.daemon = True  # 设为守护线程
        publish_thread.start()
         
        serve(data_dir,mqtt_client)

    except KeyboardInterrupt:
        print("\n程序终止")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        mqtt_client.disconnect()
        mqtt_client.loop_stop()
        print("MQTT连接已关闭")