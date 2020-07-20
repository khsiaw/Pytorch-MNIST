import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data

# Data loading
train_data = torchvision.datasets.MNIST(root='E:\Coding\Jupyter codes\Learning basics of packages\Pytorch\data',
                                        train=True, transform=transforms.ToTensor())

test_data = torchvision.datasets.MNIST(root='E:\Coding\Jupyter codes\Learning basics of packages\Pytorch\data',
                                       train=False, transform=transforms.ToTensor())

# Hyper-parameters
input_size = 784
hidden_size = 100
output_size = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=True)


# print(len(train_loader), len(test_loader))

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


# Model
model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Iterations per epoch
n_iterations = len(train_loader)

# training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)

        # forward psss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, steps {i + 1}/ {n_iterations}, loss = {loss}')

# Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader)

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)
        test_outputs = model(images)

        _, predictions = torch.max(test_outputs, -1)
        n_correct = (predictions == labels).sum().item()

    acc = n_correct / n_samples
    print(f'accuracy: {acc}')

_, predictions = torch.max(model(images), -1)
print(predictions, '\n', labels)
