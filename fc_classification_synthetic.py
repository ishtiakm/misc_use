import torchvision
from model import FC
import numpy as np
import torch
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors

# for this assignment, using a cpu should be sufficient if you don't have a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_decision_boundary(net, features, labels):
    pass


def gen_data_square(num_data, bounds):
    pass


def gen_circle_data(num_data, center_0, center_1, radius):
    pass


# slightly different from MNIST/CIFAR10 because you have no loader anymore
def test(net, data, target, device):
    pass


# slightly different from MNIST/CIFAR10 because you have no loader anymore
def train(net, training_features, training_labels, optimizer, epoch, device):
    pass


if __name__ == '__main__':

    # set hyper-parameters
    n_epochs = 5
    learning_rate = 1e-2
    seed = 100
    input_dim = 2
    out_dim = 2
    num_hidden_layers = 2
    layer_size = 50
    momentum = 0.9

    # YOUR CODE GOES HERE