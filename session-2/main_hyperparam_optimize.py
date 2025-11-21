import torch
import ray
from ray import tune
import tempfile


from dataset import MyDataset
from model import MyModel
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from ray.train import Checkpoint, get_checkpoint
from pathlib import Path
import ray.cloudpickle as pickle
from ray import train



def load_datasets(config):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    full_dataset = MyDataset(config["images_path"], config["labels_path"], transform=transform)

    total_size = len(full_dataset)
    train_size = int(config["size_train"] * total_size)
    val_size = int(config["size_eval"] * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    return train_ds, val_ds, test_ds

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
    total = 0
    correct = 0
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
    avg_loss = running_loss / total
    acc = correct / total
    return acc, avg_loss


def train_model(config):
    # Dataset
    train_dataset, val_dataset, _ = load_datasets(config)
    # Model
    my_model = MyModel(config['features'], config['hidden_layers'], config['outputs'])
    # Parallels training:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            my_model = nn.DataParallel(my_model)
    my_model.to(device)    
    
    # Dataloaders
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_model.parameters(), config["lr"])

    #check point:
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            my_model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(my_model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
             val_acc, val_loss = eval_single_epoch(my_model, val_loader, criterion, device)

    checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": my_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            {"loss": val_loss, "accuracy": val_acc},
            checkpoint=checkpoint,
        )

    print("Finished Training")

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
        "epochs": tune.choice([100]),
        "batch_size": tune.choice([64]),
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
        metric="loss",
        mode="min",
        num_samples=5,
        config = config
        )
    
    
    
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model  = MyModel(config['features'], best_trial.config['hidden_layers'], config['outputs']).to(device)
    
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    train_dataset, val_dataset, test_dataset = load_datasets(best_trial.config)
    test_acc = test_model(best_trial.config, best_trained_model, test_dataset, device)
    print("Best trial test set accuracy: {}".format(test_acc))