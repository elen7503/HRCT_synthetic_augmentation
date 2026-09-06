import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *
from data_helpers import *


def train(model,train_loader, val_loader, num_epochs, optimizer, criterion, scheduler=None, early_stopping_patience=None, plot=True, device='cpu', **kwargs):
    """
    Trains a PyTorch model using the given train and validation dataloaders for the specified number of epochs.
    """
    Net=model()
    net=Net.to(device)
    if early_stopping_patience is not None:
        best_val_loss = float('inf')
        counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
       
        running_loss = 0.0
        epoch_loss = 0.0
        total = 0
        correct = 0
        net.train()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view_as(predicted)).sum().item()
            train_acc = 100 * correct / total
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 9:
                running_loss = 0.0

        train_acc = 100 * correct / total

        net.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device).long()
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.view_as(predicted)).sum().item()

        train_loss = epoch_loss / len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        print('[Epoch %d] Train Loss: %.3f, Val Loss: %.3f, Train Acc: %.2f %%, Val Acc: %.2f %%' %
              (epoch + 1, train_loss, val_loss, train_acc, val_acc))

        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    print("Early stopping after epoch ", epoch + 1)
                    break
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)


        if scheduler is not None:
            scheduler.step()

    if plot:
        plt.plot(range(1, epoch+2), train_accs, label="Train")
        plt.plot(range(1, epoch+2), val_accs, label="Validation")
        plt.title("Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.show()

        plt.plot(range(1, epoch+2), train_losses, label="Train")
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return train_losses, val_losses, train_accs, val_accs

