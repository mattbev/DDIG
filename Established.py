#!/usr/bin/env python
# coding: utf-8

# # Testing of HPool Layer on Established Model

# ## Setup

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from Abstract import Histogram as AHist

def restrict_GPU_pytorch(gpuid, use_cpu=False):
    """
        gpuid: str, comma separated list "0" or "0,1" or even "0,1,3"
    """
    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

        print("Using GPU:{}".format(gpuid))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")
        
        
gpuid = "0"
use_cpu = True
restrict_GPU_pytorch(gpuid, use_cpu=use_cpu)


# ## Data Loading

from torch.utils.data.sampler import SubsetRandomSampler

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

#Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
def get_train_loader(batch_size):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
        return train_loader
    
#Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples+n_val_samples, dtype=np.int64))
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

#Testing
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)


# ## Load Models

def initialize_model(model_name, num_classes, resume_from = None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # The model (nn.Module) to return
    model_ft = None
    # The input image is expected to be (input_size, input_size)
    input_size = 0
    
    # You may NOT use pretrained models!! 
    use_pretrained = False
    
    # By default, all parameters will be trained (useful when you're starting from scratch)
    # Within this function you can set .requires_grad = False for various parameters, if you
    # don't want to learn them
    class Flatten(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return x.view(batch_size, -1)

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
#         dropout = nn.Dropout()
#         flatten = Flatten()
#         hist = AHist.HistPool(num_bins=10)
#         layers = list(model_ft.children())
#         layers.insert(-1, hist)
#         del layers[-1]
#         del layers[-2]
#         model_ft = nn.Sequential(*layers)
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        print(model_ft.children())
        input_size = 224
    
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        # dropout = nn.Dropout(p=.2)
        # flatten = Flatten()
        # layers = list(model_ft.children())
        # layers.insert(-1, flatten)
        # layers.insert(-1, dropout)
        # del layers[-1]
        # model_ft = nn.Sequential(*layers)
        # print(list(model_ft.children()))
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        dropout = nn.Dropout()
        flatten = Flatten()
        layers = list(model_ft.children())
        layers.insert(-1, flatten)
        layers.insert(-1, dropout)
        del layers[-1]
        model_ft = nn.Sequential(*layers)
        # print(list(model_ft.children()))
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    else:
        raise Exception("Invalid model name!")
    
    if resume_from is not None:
        print("Loading weights from %s" % resume_from)
        x = torch.load(resume_from)
        model_ft.load_state_dict(x)
    
    return model_ft, input_size

# initialize_model("resnet18", 10)


# ## Loss and Optimizer

import torch.optim as optim

def loss_optimizer_scheduler(net, learning_rate=0.001):
    """
    Initializes the loss optimizer functions
    """
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)
    
    return loss, optimizer, scheduler


# ## Training

import time
import matplotlib.pyplot as plt

def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size =", batch_size)
    print("epochs =", n_epochs)
    print("learning_rate =", learning_rate)
#     print("num_bins =", list(model.children())[8].num_bins)
    print("=" * 30)
    
    validation_losses = []
    training_losses = []
    
    #Retrieve training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Initialize loss and optimizer functions
    loss, optimizer, scheduler = loss_optimizer_scheduler(net, learning_rate)
    
    
    training_start_time = time.time()
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
#         print_every = 1 if print_every == 0 else print_every
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in tqdm(enumerate(train_loader, 0)):
            
            #Get inputs
            inputs, labels = data
#             if epoch == 0:
#                 global img
#                 img = inputs
#                 target = inputs[0]
#                 imshow(target)
#                 print(classes[labels[0]])
            inputs, labels = Variable(inputs), Variable(labels)   
                        
            #Set parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)            
            loss_size = loss(outputs, labels)
            
            #Intermediate accuracy
#             _, predicted = torch.max(outputs.data, dim=1)
#             num_correct = (predicted == labels).sum().item()
#             intermediate_acc = (num_correct * 100.0 / labels.size(0))
            
            loss_size.backward()
            optimizer.step()
            
            #Update statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            
            
            #Print statistics every 10th batch of epoch
            if (i+1) % (print_every+1) == 0:
                print("Epoch {epoch}, {percent_complete_epoch:d}% \t train_loss: {train_loss:.2f} \t took: {time:.2f}s".format(
                    epoch = epoch+1, 
                    percent_complete_epoch = int(100 * (i+1) / n_batches), 
                    train_loss = running_loss / print_every, 
                    time = time.time() - start_time, 
#                     accuracy = intermediate_acc.item()
                ))
                
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #After each epoch, run a pass on validation set
        total_val_loss = 0
        
        for inputs, labels in val_loader:
            
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
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
    


import json

model_name = "resnet18"
num_classes = 10
resume_from = None
model, input_size = initialize_model(model_name = model_name, num_classes = num_classes, resume_from = resume_from)

model = model.to(device)

batch_size = 32
n_epochs = 5
learning_rate = 1e-3

trainNet(model, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)

torch.save(model, "models/TEST")

