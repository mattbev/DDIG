import torch
import torch.nn.functional as F

from HPool import HPool

class CNN(torch.nn.Module):

    """
    Batch shape of x: (3, 32, 32)
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.fc = torch.nn.Linear(4096, 10)



    def forward(self, x):
        """
        Computes activation of the first convolution
        """

        x = F.relu(self.conv1(x))

        x = F.relu((self.pool1(x)))

        x = F.relu(self.conv2(x))

        x = F.relu((self.pool2(x)))

        _, c, h, w = x.shape
        x = x.view(-1, c*h*w)
        x = self.fc(x)

        return x



class HistNN(torch.nn.Module):
    def __init__(self):
        super(HistNN, self).__init__()






'''
class SimpleCNN(torch.nn.Module):

    # Batch shape of x: (3, 32, 32)

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input channels = 3, Output channels = 18
        conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv1.to(device)

        # Channels = 18, Height = Width = 32
        pool = HPool(18, num_bins=8)
        self.pool = pool.to(device)

        # 18432 input features, 64 output features
        fc1 = torch.nn.Linear(18, 10)
        self.fc1 = fc1.to(device)

        # 18 input features, 10 output features (for 10 defined classes)
        # fc2 = torch.nn.Linear(64, 10)
        # self.fc2 = fc2.to(device)

    def forward(self, x):
        """
        Computes activation of the first convolution
        """
        # Dimension changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Dimension changes from (18, 32, 32) to (18)
        x = F.relu(self.pool(x))

        # Dimension changes from (18) to (64)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)

        # Dimension changes from (64) to (10)
        # x = self.fc2(x)
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
        # pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.pool = pool.to(device)

        self.pool = HPool(18, num_bins=16*16)

        # 18 input features, 10 output features (for 10 defined classes)
        fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        # fc1 = torch.nn.Linear(18, 64)
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
        x = self.pool.forward(x)

        print(x.shape)
        x = x.view(-1, 18 * 16 * 16)
        print(x.shape)

        # Dimension changes from (1,18) to (1, 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def output_size(in_size, kernel_size, stride, padding):
        """
        Determines the output size
        """
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1

        return output

        
'''