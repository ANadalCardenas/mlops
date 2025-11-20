import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(my_model, dataloader, criterion, optimizer):
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)        
        y_ = my_model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()


def eval_single_epoch(my_model,  dataloader):
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_ = my_model(x)
        preds = torch.argmax(y_, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    accuracy = correct / total
    return  accuracy

def train_model(config):
    
    # Convert an image to a tensor
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
    my_dataset = MyDataset(config["images_path"], config["labels_path"], transform=transform)
    dataloader = DataLoader(my_dataset, batch_size=config['batch_size'], shuffle = True)
    my_model = MyModel(config['features'], config['hidden_layers'], config['outputs']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(my_model, dataloader, criterion, optimizer)
        with torch.no_grad():
            print(eval_single_epoch(my_model, dataloader))

    return my_model
    
    


if __name__ == "__main__":

    config = {
        "epochs": 5,
        "batch_size": 15,
        "images_path": "/home/anadal/workspace/aidl-2025-winter-mlops/session-2/archive/data/data/",
        "labels_path": "/home/anadal/workspace/aidl-2025-winter-mlops/session-2/archive/chinese_mnist.csv",
        "features" : 4096,
        "hidden_layers": 64,
        "outputs": 15,
        "lr" : 0.1

    }
    train_model(config)
