# Usage
# python train.py

# import the necessary package
from mymodule import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch


def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shaoe[0], batchSize):
        # yield a tuple of the current batched data and labels
        yield inputs[i:i + batchSize], targets[i:i + batchSize]


# specify our batch size, number of epochs and learning rate
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}..".format(DEVICE))

# generate a 3-class classification problem with 1000 data points,
# where each data point is a 4D feature vector
print("[INFO] preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, cluster_std=2.5, random_state=95)

# create training and testing splits, and convert them to PyTorch tensors
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
