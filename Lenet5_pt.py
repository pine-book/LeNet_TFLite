import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
import time


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            #nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.softmax(dim=1)
        return x


model = LeNet5()
model = torch.load("../LeNet5.pt")
model.eval()

"""input = torch.randn(1,1,28,28, dtype=torch.float)"""

input_data = np.load("../mnist_sample7.npy")
input_data = input_data.reshape(1, 1, 28, 28).astype('float32')
input_data /= 255.0

input = torch.from_numpy(input_data)

time_sta = time.perf_counter() # Timer start
for i in range(10000):
    result = model(input)
time_end = time.perf_counter() # Timer stop
tim = time_end- time_sta

print(result)
#print(result.softmax(dim=1))
"""
x = torch.randn(3, dtype=torch.float)
print(x)
y = x.softmax(dim=0)
print(y)"""
 
print(tim)



  