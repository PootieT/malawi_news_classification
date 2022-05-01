import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms 
from torch.optim import SGD, Adam
from tqdm import tqdm 


class NeuralNetwork(nn.Module):
  def __init__(self, input_size, output_size):
    super(NeuralNetwork, self).__init__()
    # You are free to define any number of layers you want to here
    self.layer1 = nn.Linear(input_size, 800) # Input Layer
    self.layer2 = nn.Linear(800, 750)        # Second Layer
    self.layer3 = nn.Linear(750, 300)        # Third Layer
    self.output = nn.Linear(300, output_size)# Output Layer
    self.dropout = nn.Dropout(0.2)


  def forward(self, input):
    x = F.relu(self.layer1(input))
    x = torch.sigmoid(self.layer2(x))
    x = torch.sigmoid(self.layer3(x))
    x = self.dropout(x)
    outputs = self.output(x)
    return F.softmax(outputs)