import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def train_single_epoch(my_model, epoch,  dataloader, criterion, optimizer):
    print(f"Epoch {epoch+1}/{config["epochs"]}")
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = my_model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()


def eval_single_epoch(my_model, epoch,  dataloader, criterion):
    print(f"Epoch {epoch+1}/{config["epochs"]}")
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_ = my_model(x)
        loss = criterion(y_, y)
        loss.backward()


def train_model(config):
    
    my_dataset = MyDataset()
    dataloader = DataLoader(my_dataset, batch_size=config['batch_size'])
    my_model = MyModel(config['features'], hidden, outputs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(my_model.parameters(), lr=args.lr)
    for epoch in range(config["epochs"]):
        train_single_epoch(my_model, epoch,  dataloader, criterion, optimizer)
        eval_single_epoch(my_model, epoch,  dataloader, criterion)

    return my_model


if __name__ == "__main__":

    config = {
        "epochs": 5,
        "batch_size": 10,
    }
    train_model(config)
