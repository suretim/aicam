import grpc
import model_pb2
import model_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = model_pb2_grpc.FederatedLearningStub(channel)

response = stub.UploadModelParams(model_pb2.ModelParams(
    weights=[0.1, 0.2, 0.3],
    client_id=1
))
