import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from HPool import HPool
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


### Helper Functions###

def GPU_init(gpuid, use_cpu=False):
    """
        gpuid: str, comma separated list "0" or "0,1" or even "0,1,3"
    """
    if not use_cpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
        dev = "cuda:{0}".format(gpuid)
        print("Using GPU:{}".format(gpuid))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        dev = "cpu"
        print("Using CPU")

    device = torch.device(dev)
    return device

def plot_loss(n_epochs, training_losses, validation_losses):
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, n_epochs, n_epochs)
    plt.plot(x, training_losses)
    plt.plot(x, validation_losses)
    ax.legend(["train_loss", "val_loss"])
    plt.draw()



### Load Data ###

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root="./cifardata", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./cifardata", train=False, download=True, transform=transform)

classes = ('plane',
           'car',
           'bird',
           'cat',
           'deer',
           'dog',
           'frog',
           'horse',
           'ship',
           'truck')


# Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))


def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    return train_loader


# Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

# Testing
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)




### CNN ###


class SimpleCNN(torch.nn.Module):

    # Batch shape of x: (3, 32, 32)

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input channels = 3, Output channels = 18
        conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv1.to(device)

        # Channels = 18, Height = Width = 32
        pool = HPool(18, 32, 32, num_bins=4)
        self.pool = pool.to(device)

        # 18432 input features, 64 output features
        fc1 = torch.nn.Linear(18, 64)
        self.fc1 = fc1.to(device)

        # 18 input features, 10 output features (for 10 defined classes)
        fc2 = torch.nn.Linear(64, 10)
        self.fc2 = fc2.to(device)

    def forward(self, x):
        """
        Computes activation of the first convolution
        """
        # Dimension changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Dimension changes from (18, 32, 32) to (18)
        x = self.pool(x)

        # Dimension changes from (18) to (64)
        x = F.relu(self.fc1(x))

        # Dimension changes from (64) to (10)
        x = self.fc2(x)
        #         sm = F.softmax(x)

        return x

    def output_size(in_size, kernel_size, stride, padding):
        """
        Determines the output size
        """
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1

        return output



class oldCNN(torch.nn.Module):

    # Batch shape of x: (3, 32, 32)

    def __init__(self):
        super(oldCNN, self).__init__()

        # Input channels = 3, Output channels = 18
        conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv1.to(device)

        # Channels = 18, Height = Width = 32
        #         self.pool = ExpPool(18, 32, 32)
        pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool = pool.to(device)

        # 18 input features, 10 output features (for 10 defined classes)
        #         self.fc = torch.nn.Linear(18, 10)
        fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc1 = fc1.to(device)

        fc2 = torch.nn.Linear(64, 10)
        self.fc2 = fc2.to(device)

    def forward(self, x):
        """
        Computes activation of the first convolution
        """
        # Dimension changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Dimension changes from (18, 32, 32) to (18, 1)
        #         x = self.pool.forward(x)
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)

        #         #Dimension changes from (1,18) to (1, 10)
        #         x = F.relu(self.fc(x))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #         sm = F.softmax(x)

        return x

    def output_size(in_size, kernel_size, stride, padding):
        """
        Determines the output size
        """
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1

        return output


### Loss/Optimizer ###

def loss_optimizer_scheduler(net, learning_rate=0.001):
    """
    Initializes the loss optimizer functions
    """
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)

    return loss, optimizer, scheduler


### Training ###


def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size =", batch_size)
    print("epochs =", n_epochs)
    print("learning_rate =", learning_rate)
    # print("num_bins =", model.pool.num_bins)
    print("=" * 30)

    validation_losses = []
    training_losses = []

    # Retrieve training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Initialize loss and optimizer functions
    loss, optimizer, scheduler = loss_optimizer_scheduler(net, learning_rate)

    training_start_time = time.time()
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # Set parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()

            # Print statistics every 10th batch of epoch
            if (i + 1) % (print_every + 1) == 0:
                print(
                    "Epoch {epoch}, {percent_complete_epoch:d}% \t train_loss: {train_loss:.2f} \t took: {time:.2f}s".format(
                        epoch=epoch + 1,
                        percent_complete_epoch=int(100 * (i + 1) / n_batches),
                        train_loss=running_loss / print_every,
                        time=time.time() - start_time,
                    ))

                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # After each epoch, run a pass on validation set
        total_val_loss = 0

        for inputs, labels in val_loader:
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

        validation_losses.append(total_val_loss / len(val_loader))
        training_losses.append(total_train_loss / len(train_loader))
        scheduler.step(total_val_loss / len(val_loader))

    print("=" * 30)
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    plot_loss(n_epochs=n_epochs,
              training_losses=training_losses,
              validation_losses=validation_losses)



### Accuracy ###

def accuracy(model):
    correct = 0
    total = 0

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    model = model.eval()

    with torch.no_grad():

        for data in test_loader:

            # Run test data through model
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)

            # Updte statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class specific
            c = (predicted == labels).squeeze()
            for i in range(len(predicted)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        # Report accuracies
        overall_acc = (correct * 100.0) / total
        class_acc = {}
        for i in range(len(classes)):
            class_acc[classes[i]] = (class_correct[i] * 100) / class_total[i]

    return overall_acc, class_acc


### Main ###
if __name__ == "__main__":

    device = GPU_init("0", use_cpu=False)

    model = oldCNN().to(device)
    batch_size = 32
    n_epochs = 1
    learning_rate = 1e-3

    trainNet(model, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)

    # torch.save(model, "models/100b100e")

    overall_acc, class_acc = accuracy(model)
    print("Final Accuracy: \n overall: {overall_acc:.3f} \n class: {class_acc}".format(
        overall_acc=overall_acc,
        class_acc=json.dumps(class_acc, indent=4)))

    plt.show()