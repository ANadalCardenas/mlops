import torch
import ray
from ray import tune


from dataset import MyDataset
from model import MyModel
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from utils import accuracy, save_model



def train_single_epoch(my_model, trainloader, criterion, optimizer, device):
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        x, y = data
        x, y = x.to(device), y.to(device)     
        y_ = my_model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()


def eval_single_epoch(my_model,  val_loader, criterion, device = "cpu"):
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(val_loader, 0):
        x, y = data
        x, y = x.to(device), y.to(device)
        y_ = my_model(x)
        loss = criterion(y_, y)
        running_loss += loss.item() * y.size(0)
        preds = torch.argmax(y_, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    accuracy = correct / total
    val_loss = running_loss / total
    return accuracy, val_loss


def train_model(config, reporter):
    device = "cpu"
    # Model
    my_model = MyModel(config['features'], config['hidden_layers'], config['outputs'])
    # Parallels training:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            my_model = nn.DataParallel(my_model)
    my_model.to(device)
    # Dataset
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
    my_dataset = MyDataset(config["images_path"], config["labels_path"], transform=transform)
    # Split dataset
    total_size = len(my_dataset)
    train_size = int(config['size_train'] * total_size)
    val_size   = int(config['size_eval'] * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, _ = random_split(my_dataset,[train_size, val_size, test_size],generator=torch.Generator().manual_seed(42))

    # Dataloaders
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_model.parameters(), config["lr"])

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(my_model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_acc, val_loss = eval_single_epoch(my_model, val_loader, criterion)
    reporter(val_loss=val_loss, val_accuracy=val_acc)


def test_model(config, model, test_dataset, device):
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


    config = {
        "epochs": tune.choice([2, 4, 6]),
        "batch_size": tune.choice([2, 4]),
        "images_path": "/home/anadal/workspace/aidl-2025-winter-mlops/session-2/archive/data/data/",
        "labels_path": "/home/anadal/workspace/aidl-2025-winter-mlops/session-2/archive/chinese_mnist.csv",
        "features" : 4096,
        "hidden_layers": tune.choice([2 ** i for i in range(9)]),
        "outputs": 15,
        "lr" : tune.loguniform(1e-2, 1e-1),
        "size_train" : 0.7,
        "size_eval" : 0.2
        }
    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config = config
        )
    
    
    best_config = analysis.best_config
    print("Best hyperparameters found were: ", best_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Dataset
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
    my_dataset = MyDataset(best_config["images_path"], best_config["labels_path"], transform=transform)
    # Split dataset
    total_size = len(my_dataset)
    train_size = int(best_config['size_train'] * total_size)
    val_size   = int(best_config['size_eval'] * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(my_dataset,[train_size, val_size, test_size],generator=torch.Generator().manual_seed(42))
    # Dataloaders
    train_loader = DataLoader(train_dataset,batch_size=best_config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=best_config['batch_size'], shuffle=False)
    # Best model
    final_model = MyModel(
        best_config["features"], best_config["hidden_layers"], best_config["outputs"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(final_model.parameters(), best_config["lr"])
    # Training:
    for epoch in range(best_config["epochs"]):
        print(f"Epoch {epoch+1}/{best_config['epochs']}")
        train_single_epoch(final_model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_acc, val_loss = eval_single_epoch(final_model, val_loader, criterion)
            print(f"[FINAL] val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
    
    test_acc = test_model(best_config, final_model, test_dataset, device)
    print(f"Test accuracy with best hyperparameters: {test_acc:.4f}")
