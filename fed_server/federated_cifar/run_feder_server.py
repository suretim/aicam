import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc
import time

class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self, mqtt_client):
        self.mqtt_client = mqtt_client  # 将 MQTT 客户端传入

    def GetModel(self, request, context):
        # 模拟返回一个模型参数
        model_weights = [0.1] * 100  # 模拟一个包含 100 个权重的模型
        print("Sending model to MQTT...")
        self.send_model_to_mqtt(model_weights)  # 将模型发送到 MQTT
        return model_pb2.ModelResponse(weights=model_weights)

    def send_model_to_mqtt(self, model_weights):
        # 将模型权重发送到 MQTT
        topic = "model/weights"
        message = {"weights": model_weights}
        self.mqtt_client.publish(topic, str(message))  # 将模型发布到 MQTT
        print("Model sent to MQTT.")

def serve():
    # 初始化 MQTT 客户端（这里使用一个假设的 MQTT 客户端）
    import paho.mqtt.client as mqtt

    mqtt_client = mqtt.Client()
    mqtt_client.connect("192.168.0.57", 1883, 60)  # 连接到 MQTT Broker

    # 创建 gRPC 服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(mqtt_client), server)

    # 启动 gRPC 服务器
    print("gRPC server is starting...")
    server.add_insecure_port('[::]:50051')
    server.start()

    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
