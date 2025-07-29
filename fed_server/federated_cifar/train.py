import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import SimpleCNN

def train(net, trainloader, epochs=1, device="cpu"):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device="cpu"):
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss / total, correct / total
