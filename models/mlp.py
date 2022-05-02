import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from network import *


epochs = 5  # number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
criterion = nn.CrossEntropyLoss()  # loss function
batch_size = 16
lr = 1e-6


class CustomizedDataset(Dataset):
    def __init__(self, X, Y):
        self.input_data = X
        self.labels = Y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_data[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    def fit(self, train_data, train_labels):

        input_size = train_data.shape[1]
        output_size = len(np.unique(train_labels))

        self.network = Network(input_size, output_size).to(device)
        optimizer = Adam(self.network.parameters(), lr=lr)
        total_loss = 0

        train_data = train_data.astype(np.float32)
        train_dataset = CustomizedDataset(train_data, train_labels)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        train_dataloader2 = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        pbar = tqdm(total=epochs, position=0, leave=True)
        for epoch in range(epochs):
            epoch_loss = 0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data

                optimizer.zero_grad()
                outputs = self.network(
                    inputs.view(inputs.shape[0], -1).to(device)
                )  # getting outputs from the network

                # labels_ = F.one_hot(labels, num_classes=output_size)

                # loss = criterion(outputs, labels_.to(device).float())
                # loss = criterion(outputs, labels_.to(device))
                loss = criterion(outputs, labels.to(device))
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
                    outputs = self.network(input.view(input.shape[0], -1).to(device))

                    total += len(labels)

                    predictions = torch.argmax(outputs, dim=1)
                    predictions = predictions.to("cpu").numpy()
                    correct += sum(1 * (labels.numpy() == predictions))

            # print(
            #     "  ---  epoch loss = %1.2f  --- training accuracy = %1.2f "
            #     % (epoch_loss, correct / total)
            # )
            pbar.update(n=1)
            pbar.set_description(f"Epoch: {epoch}/{epochs}, Loss: {epoch_loss:.6f}, Accuracy: {correct / total:.6f}")

    def predict(self, test_data):
        batch_size = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device

        test_labels = np.zeros(len(test_data))
        test_data = test_data.astype(np.float32)
        # print("test data shape = ", test_data.shape)
        test_dataset = CustomizedDataset(test_data, test_labels)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        preds = []
        with torch.no_grad():
            self.network.eval()
            total = 0
            for i, data in enumerate(test_dataloader):
                input, labels = data
                outputs = self.network(input.view(input.shape[0], -1).to(device))

                total += len(labels)

                predictions = torch.argmax(outputs, dim=1)
                predictions = predictions.to("cpu").numpy()
                preds.append(predictions)

        preds = np.array([item for sublist in preds for item in sublist])
        return preds
