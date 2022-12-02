from typing import Tuple, List, Dict  ## used to define types used in tuple, list, dict
import numpy as np
import pandas

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_url_detector import URL_Neural_Net
from timeit import default_timer as timer  ## use to keep track of training time
from tqdm.auto import tqdm ## pretty prints and progress bar for training

from train_url_detector import URLDataset
import sklearn.utils   ## used to shuffle the entries for dataset



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ## defines how many sample to analyse per category
    NUM_OF_SAMPLES = 3
    
    
    CLASS_LABELS = {'defacement': 0, 'benign': 1, 'phishing': 2, 'malware': 3}
    model = URL_Neural_Net(input_shape=1, hidden_units=256, output_shape=len(CLASS_LABELS))
    model.load_state_dict(torch.load("./models/model.pth"))
    model = model.to(device)
    rawdata = pandas.read_csv("malicious_phish.csv")
    
   
    MAX_URL_LENGTH = rawdata.url.str.len().max()
    print(f"Class labels: {CLASS_LABELS}")
    
    
    key_list = list(CLASS_LABELS.keys())
    val_list = list(CLASS_LABELS.values())
    
    print("################### beginning analysis... #####################")
    model.eval()
    LABELS = list(CLASS_LABELS.keys())
    for i in range(len(LABELS)):
        targets = rawdata.loc[rawdata["type"] == LABELS[i]]
        targets = targets.sample(frac=1).reset_index(drop=True)
        print(f"\t\tAnalysis for label: {LABELS[i]}")
        for j in range(NUM_OF_SAMPLES):
            url, label = targets.iloc[j]["url"], targets.iloc[j]["type"]
            print(f"URL: {url}")
            url = preprocess_text(url, max_length=MAX_URL_LENGTH)
            url = url.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                y_pred = model(url)
            print(f"\tModel pred: \t{y_pred.argmax(dim=1).item()} - {key_list[val_list.index(y_pred.argmax(dim=1).item())]} with confidence { [np.round(i, 3) for i in torch.round(torch.softmax(y_pred, dim=1), decimals=3).tolist()]}")
        print("")
            
        
def preprocess_text(target: str, max_length:int):
    """This function preprocesses raw text input and returns a tensor containing the input text in ascii format"""
    ascii = [ord(c) for c in target]
    return nn.ConstantPad1d((0, max_length-len(ascii)), 0)(torch.Tensor(ascii))

def convert_ascii_to_url(target):
    lst = target.squeeze().tolist()
    return ''.join([chr(int(c)) if c != 0 else '' for c in lst])

    
if __name__ == "__main__":
    main()