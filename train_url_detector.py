from typing import Tuple, List, Dict  ## used to define types used in tuple, list, dict
import numpy as np
import os
import random
import pandas

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import torchmetrics

import pickle
from timeit import default_timer as timer  ## use to keep track of training time
from tqdm.auto import tqdm ## pretty prints and progress bar for training


import sklearn.utils   ## used to shuffle the entries for dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rawdata = pandas.read_csv("malicious_phish.csv")
    print("Dataset loaded successfully, peeking at some entries...")
    print(rawdata)  ## lets see what the pandas.dataframe look like. 
    print(rawdata.iloc[0]["url"])
    
    print("#####################################################")
    
    dataset = URLDataset()
    MAX_URL_LENGTH = dataset.get_max_url_length()
    CLASS_LABELS = dataset.get_labels_with_index()

    random_index = random.randint(0, len(dataset))

    print(f"At index {random_index}, the url is {dataset[random_index][0]} and the type is {dataset[random_index][1]}")
    url, label = rawdata.iloc[random_index]["url"], rawdata.iloc[random_index]["type"]
    print(f"At rawid {random_index}, the url is {url} and the type is {label}")
    print(f"The maximum URL length is: {MAX_URL_LENGTH}")
    print(f"The indexes are {CLASS_LABELS}")


    print("#####################################################")
    # {'phishing': 0, 'benign': 1, 'defacement': 2, 'malware': 3}

    print(f"The total length of the dataset used for training and testing is {len(dataset)}")
    
    ### split the dataset into training and testing sets
    split = 0.8  ## can be changed depending on what you want
    train_size = int(split*len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])


    print("Starting data exploration...")
    start_time = timer()
    ## 1. DATAEXPLORATION: obtain some distribution stats
    total_distribution = torch.zeros(len(CLASS_LABELS), dtype=torch.long)
    for _, target in dataset:
        total_distribution[target] += 1
        

    train_distribution = torch.zeros(len(CLASS_LABELS), dtype=torch.long)
    for _, target in train_data:
        train_distribution[target] += 1
        
    test_distribution = torch.zeros(len(CLASS_LABELS), dtype=torch.long)
    for _, target in test_data:
        test_distribution[target] += 1

    print(f"For the entire dataset (count and percent):")
    total_count, total_percent = get_distributions(total_distribution, CLASS_LABELS, len(dataset))
    print(total_count)
    print(total_percent)

    print(f"\nFor the training dataset (count and percent):")
    train_count, train_percent = get_distributions(train_distribution, CLASS_LABELS, len(train_data))
    print(train_count)
    print(train_percent)

    print(f"\nFor the testing dataset (count and percent):")
    test_count, test_percent = get_distributions(test_distribution, CLASS_LABELS, len(test_data))
    print(test_count)
    print(test_percent)
    
    print(f"The time it took to analyse the statistics: {timer()-start_time}")
    
    
    ## load the data into dataloader
    BATCH_SIZE = 128
    NUM_WORKER = min(1, os.cpu_count()-2)
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)


    url, label = next(iter(train_dataloader))
    print(f"The shape of the inputs after dataloader is {url.shape} and the label is {label.shape}")
    
    print(f"URL is in the form {url[0]}, converted: {convert_ascii_to_url(url[0])}")
    
    model = URL_Neural_Net(input_shape=1, hidden_units=256, output_shape=len(CLASS_LABELS)).to(device)
    
    print(model)
    
    print("------------------MODEL SUMMARY ---------------------")
    print(summary(model, (1, 2175)))
    
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(params=model.parameters(), lr=0.001)
    accuracy_fn = torchmetrics.Accuracy().to(device)


    start_time = timer()

    epochs = 15
    ## Used to plot training curves
    train_loss_hist, train_acc_hist = [], []
    test_loss_hist, test_acc_hist = [], []

    print("Starting model training...")
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in (enumerate(train_dataloader)):
            # print(batch)
            # start_time = timer()
            X, y = X.to(device), y.to(device)
            # print(f"It took {timer() - start_time} seconds to load the data to gpu")
            X = X.unsqueeze(dim=1)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_pred.argmax(dim=1), y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}")

        """Model evaluation to see how we are doing every epoch"""
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(dim=1)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss
                test_acc += accuracy_fn(y_pred.argmax(dim=1), y)
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")  
    
    print(f"It took {timer() - start_time} seconds to train the model!")
    
    
    
    key_list = list(CLASS_LABELS.keys())
    val_list = list(CLASS_LABELS.values())
    
    """Example output"""
    X, y = next(iter(test_dataloader))
    for i in range(10):
        url, label = X[i], y[i]
        url, label = url.to(device), label.to(device)
        url = url.unsqueeze(dim=0)
        url = url.unsqueeze(dim=0)
        
        model.eval()
        with torch.inference_mode():
            y_pred = model(url)
            
    """save the model"""
    print(f"The indexes are {CLASS_LABELS}")
    torch.save(model.state_dict(), "./models/model.pth")
    
    """dump the test_set for further analysis in predict_url.py"""
    file = open("./training_hist/train_hist.pickle", "wb")
    pickle.dump(
        dict(
            train_data=train_data,
            test_data=test_data,
            train_loss_hist=train_loss_hist,
            train_acc_hist=train_acc_hist,
            test_loss_hist=test_loss_hist,
            test_acc_hist=test_acc_hist
        ), 
        file)
    file.close()     
    
    """Lets print some sample output to see what is happening"""
    print(f"URL: {convert_ascii_to_url(url)}")
    print(f"\tGround truth: \t{key_list[val_list.index(label.item())]} - {label.item()}")
    print(f"\tModel pred: \t{y_pred.argmax(dim=1).item()} - {key_list[val_list.index(y_pred.argmax(dim=1).item())]}")
        
        # print(f"For the url {convert_ascii_to_url(url)} with ground truth {label}, the model predicted: {y_pred.argmax(dim=1).item()}")
    

        
    
    
def get_distributions(distribution, labels, datatset_length):
    """returns the count and percent of the data distributions for printing"""
    count = {list(labels.keys())[i]: int(distribution[i]) for i in range(len(distribution))}
    percent = {list(labels.keys())[i]: 
    round(int(100*distribution[i]) / datatset_length, 2) for i in range(len(distribution))}
    return count, percent



def preprocess_text(target: str, max_length:int):
    """This function preprocesses raw text input and returns a tensor containing the input text in ascii format"""
    ascii = [ord(c) for c in target]
    return nn.ConstantPad1d((0, max_length-len(ascii)), 0)(torch.Tensor(ascii))

def convert_ascii_to_url(target):
    """This function returns ascii representation in a tensor to string"""
    lst = target.squeeze().tolist()
    return ''.join([chr(int(c)) if c != 0 else '' for c in lst])

class URL_Neural_Net(nn.Module):
  """Lets create a neural network"""
  def __init__(self, input_shape:int, hidden_units: int, output_shape:int) -> None:
    super().__init__()
    """Conv block 1, uses max pooling"""
    self.conv1 = nn.Sequential(
        nn.Conv1d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2)
    )
    """Conv block 2, no pooling"""
    self.conv2 = nn.Sequential(
        nn.Conv1d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=0),
        nn.ReLU()
    )

    """Flatten into a 1d in_feature, number is calculated by running it once. the model then passes through a dense layer and a dropout layer"""
    self.linear1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=277760, out_features=1024),
        nn.ReLU(),
        nn.Dropout(0.5)
    )

    """Simple layer to output_shape for classification."""
    self.classifier = nn.Linear(in_features=1024, out_features=output_shape)


  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.linear1(x)
    x = self.classifier(x)
    return x

    
class URLDataset(torch.utils.data.Dataset):
    """Create a new Dataset manager for malicious URL detection"""
    def __init__(self) -> None:
        self.data = pandas.read_csv("malicious_phish.csv")
        self.MAX_LENGTH = self.data.url.str.len().max()
        self.shuffle()  ## MUST SHUFFLE THE DATA, IT IS SOMEWHAT SORTED BY DEFAULT
        
        # self.data = self.data.head(int( len(self.data) /15))   ### split the dataset into smaller pieces to reduce train time if needed for testing
        
        self.classes_with_idx = {class_name: i for i, class_name in enumerate(self.data['type'].unique())}
        # self.classes = self.data["type"].unique()
        """README: In future runs, please use sorted() so that the output label is always consistent (otherwise it will be random based on order of appearance from shuffle. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html"""
        self.classes = sorted(self.data["type"].unique())
        
    def __len__(self) -> int:
        """returns the total number of data in the csv"""
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """returns a url and an index"""
        ascii = [ord(c) for c in self.data.iloc[index]["url"]]
            
        return nn.ConstantPad1d((0, self.MAX_LENGTH-len(ascii)), 0)(torch.Tensor(ascii)), self.classes_with_idx[self.data.iloc[index]["type"]]
    
    def get_url_by_index(self, index: int):
        return self.data.iloc[index]["url"]
    
    def get_labels(self):
        return self.classes
    
    def get_labels_with_index(self):
        return self.classes_with_idx
    
    def shuffle(self):
        """Call this method to shuffle the dataframe"""
        self.data = sklearn.utils.shuffle(self.data)
    
    def get_max_url_length(self):
        return self.MAX_LENGTH
    
    ### {'phishing': 0, 'benign': 1, 'defacement': 2, 'malware': 3}
    
if __name__ == "__main__":
    main()