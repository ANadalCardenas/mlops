import torch
from ray import tune

from dataset import MyDataset
from model import MyModel
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from utils import accuracy, save_model

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


def train_model(config, train_dataset, val_dataset):

    # Dataset
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
    my_dataset = MyDataset(config["images_path"], config["labels_path"], transform=transform)
    # Split dataset
    total_size = len(my_dataset)
    train_size = int(config['size_train'] * total_size)
    val_size   = int(config['size_eval'] * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(my_dataset,[train_size, val_size, test_size],generator=torch.Generator().manual_seed(42))

    # Dataloaders
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    my_model = MyModel(config['features'], config['hidden_layers'], config['outputs']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_model.parameters(), config["lr"])

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(my_model, train_loader, criterion, optimizer)
        with torch.no_grad():
            print(f"Acuracy: {eval_single_epoch(my_model, val_loader)}")



def test_model(config, model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            preds = torch.argmax(y_, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        accuracy = correct / total
    return  accuracy


if __name__ == "__main__":



    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            "hyperparam_1": tune.uniform(1, 10),
            "hyperparam_2": tune.grid_search(["relu", "tanh"]),
        })
    
    

    print("Best hyperparameters found were: ", analysis.best_config)
    print(test_model(...))
