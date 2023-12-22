import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
import sys
import argparse
sys.path.append('src')
from config import *
from torch import nn
from model import VGG
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
np.random.seed(0)


# create train loop and test loop
def train_loop(model, optimizer, criterion, train_loader):
    """
        Training loop for a neural network.

        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (torch.nn.Module): The loss function.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.

        Returns:
            tuple: Tuple containing the total loss, number of correct predictions, and total samples processed.
        """
    model.train()
    total_loss = 0.0
    correct_predictions = 0.0
    total_samples = 0

    for i, data in enumerate(train_loader):
        X = data[0].to(DEVICE)
        y = data[1].unsqueeze(1).to(torch.float32).to(DEVICE)

        # Forward pass
        pred = model(X)
        loss = criterion(pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        correct = ((pred > 0.5) == y).float().sum()
        correct_predictions += correct.item()
        total_samples += len(y)

        # Accumulate total loss for the epoch
        total_loss += loss.item()

    # Calculate average loss and accuracy for the epoch
    average_loss = total_loss / len(train_loader)
    accuracy = (correct_predictions / total_samples) * 100

    print(f"Train --> Loss/epoch: {average_loss}, Accuracy/epoch: {accuracy}%")
    return average_loss, accuracy
def val_loop(model, criterion, val_loader):
    """
        Validation loop for a neural network.

        Args:
            model (torch.nn.Module): The neural network model.
            criterion (torch.nn.Module): The loss function.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            tuple: Tuple containing the total loss, number of correct predictions, and total samples processed.
        """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X = data[0].to(DEVICE)
            y = data[1].unsqueeze(1).to(torch.float32).to(DEVICE)

            # Forward pass
            pred = model(X)
            loss = criterion(pred, y)

            # Calculate accuracy
            correct = ((pred > 0.5) == y).float().sum()
            correct_predictions += correct.item()
            total_samples += len(y)

            # Accumulate total loss for the validation set
            total_loss += loss.item()

    # Calculate average loss and accuracy for the validation set
    average_loss = total_loss / len(val_loader)
    accuracy = (correct_predictions / total_samples) * 100

    print(f"Validation --> Loss/epoch: {average_loss}, Accuracy/epoch: {accuracy}%")
    return average_loss, accuracy
def main(args):
    """
        Main function for training a neural network on a custom dataset.

        Args:
            args (dict): Dictionary containing command-line arguments or default values.
                Possible keys: 'data_path', 'saved_model_path', 'image_height', 'image_width',
                               'epoch', 'lr', 'batch_size'.

        Returns:
            None
        """
    dataset_path = args["data_path"] if args["data_path"] else DATA_PATH
    save_model_path = args["saved_model_path"] if args["saved_model_path"] else SAVED_MODEL_FOLDER

    image_height = args["image_height"] if args["image_height"] else IMAGE_SIZE[0]
    image_width = args["image_width"] if args["image_width"] else IMAGE_SIZE[1]
    epochs = args['epoch'] if args['epoch'] else EPOCHS
    lr = args['lr'] if args['lr'] else LR
    batch_size = args['batch_size'] if args['batch_size'] else BATCH_SIZE
    # Load and transform data
    transform = transforms.Compose([transforms.Resize((image_height, image_width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    pizza_not_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    train_ind, val_ind = train_test_split(list(range(len(pizza_not_dataset.targets))), shuffle=True, test_size=0.2,
                                           stratify=pizza_not_dataset.targets)
    train_ind, test_ind = train_test_split(train_ind,  shuffle=True, test_size=0.2, stratify=[pizza_not_dataset.targets[i] for i in train_ind])
    train_dataset = torch.utils.data.Subset(pizza_not_dataset, train_ind)
    val_dataset = torch.utils.data.Subset(pizza_not_dataset, val_ind)
    test_dataset = torch.utils.data.Subset(pizza_not_dataset, test_ind)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # h. parameters
    model = VGG()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Set experiment name and ports
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("Metrics")
    # Training loop with MLflow tracking
    with mlflow.start_run() as run:
        # train and test model
        for i in range(epochs):
            train_loss, train_accuracy = train_loop(model, optimizer, criterion, train_loader)
            val_loss, val_accuracy = val_loop(model, criterion, val_loader)
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("train_accuracy", train_accuracy, step=i)

            mlflow.log_metric("val_loss", val_loss, step=i)
            mlflow.log_metric("val_accuracy", val_accuracy, step=i)
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{save_model_path}/vgg_{i}_epoch.pth')
    val_loop(model, criterion, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help='Specify path to your dataset')
    parser.add_argument("--saved_model_path", type=str,
                        help='Specify path for save models, where models folder will be created')
    parser.add_argument("--epoch", type=int,
                        help='Specify epoch for model training')
    parser.add_argument("--batch_size", type=int,
                        help='Specify batch size for model training')
    parser.add_argument("--lr", type=float,
                        help='Specify learning rate')
    parser.add_argument("--image_height", type=float,
                        help='Specify image height')
    parser.add_argument("--image_width", type=float,
                        help='Specify image width')
    args = parser.parse_args()
    args = vars(args)
    main(args)