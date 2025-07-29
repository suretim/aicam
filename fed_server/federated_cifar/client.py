import flwr as fl
import torch
from torch.utils.data import DataLoader
from utils import load_data, SimpleCNN
from train import train, test

class CifarClient(fl.client.Client):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainset, testset = load_data()
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=32)
        self.model = SimpleCNN().to(self.device)

    def get_parameters(self, config):
        # 返回一个包含 'parameters' 键的字典，解决错误
        return {
            "parameters": [val.cpu().numpy() for val in self.model.state_dict().values()]
        }

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters["parameters"]):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main():
    client = CifarClient()
    fl.client.start_client(server_address="localhost:8081", client=client)

if __name__ == "__main__":
    main()
