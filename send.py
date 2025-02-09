import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodel_simple import ComplexCNN as CNN########
from matplotlib import pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import math
import numpy as np
import os
import argparse

import pickle

from utils import progress_bar
from utils import get_config
import os
from myMNIST import MNISTDataset
from moe_SE import MoE##
# Set CUDA devices to GPUs 5, 6 and 7
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

#torch.cuda.set_device(0)
torch.manual_seed(1)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


config = get_config()
batch_size = config['batch_size']
num_classes=config['classes']
patch_size = config['patch_size']
num_experts = config['experts']
top_patches=config['top_patches']
num_epochs=config['epochs']
strategy=config['strategy']
lrs=config['learning_rates']
regs=config['regularization']
patience=config['patience_ES']
model_savefile=config['model_savename']
k=1
if strategy=='topk':
    k=config['top_experts']
    if k> num_experts:
        print(f"K({k}) is bigger than total number of experts({num_experts})... reducing k to {num_experts}")
        k=num_experts

def train_moe_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=10, device='cuda', patience=5):
    """
    Function to train the MoE model with learning rate scheduler and early stopping.

    Args:
    - model: The MoE model.
    - trainloader: The DataLoader for the training set.
    - testloader: The DataLoader for the validation/test set.
    - criterion: Loss function (e.g., CrossEntropyLoss).
    - optimizer: Optimizer (e.g., Adam).
    - scheduler: Learning rate scheduler.
    - num_epochs: Number of training epochs.
    - device: Device to run the model on ('cuda' or 'cpu').
    - patience: Number of epochs to wait for improvement before stopping.

    Returns:
    - model: The trained model.
    """
    model.train()  # Set the model to training mode
    best_loss = float('inf')
    best_train_acc=-float('inf')
    patience_counter = 0
    
    train_acc=[]
    test_acc=[]

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training loop
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            for opt in optimizer:
                opt.zero_grad()

            # Forward pass
            outputs = model(inputs)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            for opt in optimizer:
                opt.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += loss.item()
            progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (running_loss/(i+1), 100.*total_correct/total_samples, total_correct, total_samples))

        # Evaluate on the validation set
        model.eval()
        test_loss, test_accuracy = test_moe_model(model, testloader, criterion, device=device,in_train=1)
        model.train()
        
        train_accuracy=1.*total_correct/total_samples
        
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        # Adjust learning rate based on validation loss
        for sch in scheduler:
            sch.step()

        print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")

        # Early Stopping Check
        if test_accuracy > best_train_acc:
            best_train_acc = test_accuracy
            patience_counter = 0
            # Save the best model state
            torch.save(model.state_dict(), "best_model_state.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. Best Training Accuracy: {best_train_acc:.4f}")
                break
            
        
        

    print('Finished Training')
    torch.save(model.state_dict(), "last_model_state.pth")
    # Load the best model before returning
    model.load_state_dict(torch.load("best_model_state.pth",weights_only=True))
    return model,train_acc,test_acc

def test_moe_model(model, testloader, criterion, device='cuda',in_train=0):
    """
    Function to test the MoE model.

    Args:
    - model: The MoE model.
    - testloader: The DataLoader for the test set.
    - criterion: Loss function (e.g., CrossEntropyLoss).
    - device: Device to run the model on ('cuda' or 'cpu').

    Returns:
    - test_loss: Average loss on the test set.
    - test_accuracy: Accuracy on the test set.
    """
    #model.to(device)
    model.eval()  # Set the model to evaluation mode

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, labels in testloader:
            inputs=inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(testloader)
    test_accuracy = 1.*correct / total
    if in_train==0:
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return test_loss, test_accuracy


# Define a transform to normalize the data
print('==> Preparing data..')


# Load MNIST synthetic data dataset
training_images = np.load('./data/Training_Data_patch_mnist_n_16.npy')
training_labels = np.load('./data/Training_Label_patch_mnist_n_16.npy')

test_images = np.load('./data/Test_Data_patch_mnist_n_16.npy')
test_labels = np.load('./data/Test_Label_patch_mnist_n_16.npy')

train_dataset = MNISTDataset(training_images, training_labels)
test_dataset = MNISTDataset(test_images, test_labels)

trainloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader=DataLoader(test_dataset, batch_size=64, shuffle=True)
# Determine input channels dynamically
#sample_image, _ = trainset[0]  # Access the first sample (image, label)
in_channels = 1
# Initialize the model, criterion, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))




test_accuracies=[]
train_accuracies=[]

moe_model = MoE(C=in_channels, d=patch_size, E=num_experts, l=top_patches, num_classes=num_classes,Expert=CNN,strategy=strategy,k=k).to(device)

total_params = count_parameters(moe_model)
print(f"Total trainable parameters: {total_params}")

moe_model = nn.DataParallel(moe_model, device_ids=device_ids)  # Parallelizing the model  # Example configuration
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(moe_model.parameters(), lr=0.001,weight_decay=weight_decay)#0.001 initial
optimizer1 = torch.optim.Adam(moe_model.module.experts.parameters(), lr=lrs[2],
        weight_decay=regs[2])
optimizer2 = torch.optim.Adam(moe_model.module.routers.parameters(), lr=lrs[1],
         weight_decay=regs[1])
optimizer3 = torch.optim.Adam(moe_model.module.srouter.parameters(), lr=lrs[0],weight_decay=regs[0])

optimizer=[optimizer1,optimizer2,optimizer3]#lr,wd:[2,2,2],[3,3,3]

scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=7000)#500
#scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=6000)
#scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=6000)

#scheduler = [scheduler1, scheduler2, scheduler3]
scheduler = [scheduler1]


# Train the model
trained_model,train_acc,test_acc = train_moe_model(
    moe_model,
    trainloader,
    testloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=num_epochs,
    device=device,
    patience=patience
)
# Train the model
#trained_model = train_moe_model(moe_model, trainloader, criterion, optimizer, num_epochs=num_epochs, device=device)
torch.save(trained_model.state_dict(), model_savefile)
# Test the model

with open("accuracy_data.pkl", "wb") as f:
    pickle.dump({"train_acc": train_acc, "test_acc": test_acc}, f)

print("Data saved to accuracy_data.pkl")

show=0

if show==1:
    plt.plot([i+1 for i in range (len(train_acc))], train_acc, 'bo')
    plt.plot([i+1 for i in range (len(train_acc))], train_acc, 'r',label='Train Accuracies')
    plt.plot([i+1 for i in range (len(test_acc))], test_acc, 'ro')
    plt.plot([i+1 for i in range (len(test_acc))], test_acc, 'b',label='Test Accuracies')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(['Epochs Vs Accuracy '])
    file_name=config['file_name']
    plt.savefig(file_name)
    plt.show()



