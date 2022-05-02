import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms 
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 
import numpy as np
from network import Network

class NeuralNetwork(nn.Module):

  ######################################
  ### Network Parameters
  def __init__(self):
    self.network = None


  def fit(self, train_data, train_labels):
    epochs = 5                                                            # number of epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # device
    criterion = nn.CrossEntropyLoss()                                         # loss function
    batch_size = 5

    self.network = Network(train_data.shape[0], len(np.unique(train_labels))).to(device)
    optimizer = Adam(network.parameters(), lr = 1e-2)

    total_loss = 0

    #######################################
    ### Downloading the data

    train_data = torch.tensor([np.array(train_data)])
    train_labels = torch.tensor([np.array(train_labels)])

    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=2, drop_last=True)
    train_dataloader2 = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=2, drop_last=True)
    ########################################
    ### Training Loop

    gradients = []
    for epoch in tqdm(range(epochs), desc = "epochs"): 
      epoch_loss = 0
      
      for i, data in enumerate(train_dataloader): 
        inputs, labels = data 

        optimizer.zero_grad() 
        outputs = self.network(inputs.view(inputs.shape[0],-1).to(device))  # getting outputs from the network

        labels_ = F.one_hot(labels, num_classes= 10)

        loss = criterion(outputs,labels_.to(device).float())
        loss.backward()  
        optimizer.step()  

        total_loss += loss.item() 
        epoch_loss += loss.item()
      
      with torch.no_grad():
        self.network.eval()
        correct = 0
        total = 0
        for i, data in enumerate(train_dataloader2):
          input, labels = data
          outputs = self.network(input.view(input.shape[0],-1).to(device))

          total+= len(labels)

          predictions = torch.argmax(outputs, dim = 1)
          predictions = predictions.to("cpu").numpy() 
          correct += sum(1*(labels.numpy()==predictions))
      
      print( "  ---  epoch loss = %1.2f  --- training accuracy = %1.2f " %( epoch_loss, correct/total))

  def score(self, test_data, test_labels):
    batch_size = 5 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # device
    test_data = torch.tensor([np.array(test_data)])
    test_labels = torch.tensor([np.array(test_labels)])

    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers=2, drop_last=True)

    preds = []
    with torch.no_grad():
      self.network.eval()
      correct = 0
      total = 0
      for i, data in enumerate(test_dataloader):
        input, labels = data
        outputs = self.network(input.view(input.shape[0],-1).to(device))

        total+= len(labels)

        predictions = torch.argmax(outputs, dim = 1)
        predictions = predictions.to("cpu").numpy() 
        preds.append(predictions)
        correct += sum(1*(labels.numpy()==predictions))
    
    print( "  test accuracy = %1.2f " %(correct/total))
    preds = np.array(preds).flatten()
    return preds 
    
