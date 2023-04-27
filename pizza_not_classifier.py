import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch import nn
from model import VGG
from sklearn.model_selection import train_test_split
from torchvision.models import vgg19
torch.manual_seed(0)
np.random.seed(0)



if not os.path.exists('../Pizza_or_not/models'):
    os.makedirs('../Pizza_or_not/models')


def show_predict(model, loader):
    batch = next(iter(loader))
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    labels = labels.unsqueeze(1)
    labels = labels.to(torch.float32)
    pred = model(images)
    images = images.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])
    mean = mean.reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225])
    std = std.reshape(1, 3, 1, 1)
    images = (images * std) + mean
    grid = torchvision.utils.make_grid(images[0:8], nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print((pred[0:8] > 0.5).float().reshape(1, 8))
    plt.show()
    plt.close()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG()
    model.to(device)

# Load and transform data
    path = '../Pizza_or_not/pizza_dataset'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    pizza_not_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    train_ind, test_ind = train_test_split(list(range(len(pizza_not_dataset.targets))), shuffle=True, test_size=0.2,
                                           stratify=pizza_not_dataset.targets)
    train_dataset = torch.utils.data.Subset(pizza_not_dataset, train_ind)
    test_dataset = torch.utils.data.Subset(pizza_not_dataset, test_ind)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
# h. parameters
    epoch = 10
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

# create train loop and test loop
    def train_loop(model, optimizer, criterion, train_loader):
        model.train()
        for i, data in enumerate(train_loader):
            X = data[0].to(device)
            y = data[1].unsqueeze(1)
            y = y.to(torch.float32)
            y = y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = ((pred>0.5) == y).float().sum()
            accuracy = (correct / len(y)) * 100
            print(f"train_loss: {loss.item()}, accuracy/batch: {accuracy}")

    def test_loop(model, criterion, test_loader):
        model.eval()
        size, correct = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                size += len(data[1])
                X = data[0].to(device)
                y = data[1].unsqueeze(1)
                y = y.to(torch.float32)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y).item()
                correct += (((pred > 0.5) == y).float().sum())
        accuracy = (correct / size) * 100
        print(f"accuracy: {accuracy}")

# train and test model
    for i in range(epoch):
        train_loop(model, optimizer, criterion, train_loader)
        test_loop(model, criterion, test_loader)
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'../Pizza_or_not/models/vgg_{i}_epoch.pth')
# show some predicted values
    show_predict(model, test_loader)