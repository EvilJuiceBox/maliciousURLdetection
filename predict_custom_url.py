import numpy as np
import torch
from torch import nn

from train_url_detector import URL_Neural_Net
from train_url_detector import URLDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MAX_URL_LENGTH = 2175
    """Hardcoded class labels, may have to change this per new model"""
    CLASS_LABELS = {'defacement': 0, 'benign': 1, 'phishing': 2, 'malware': 3}
    
    model = URL_Neural_Net(input_shape=1, hidden_units=256, output_shape=len(CLASS_LABELS))
    model.load_state_dict(torch.load("./models/model.pth"))
    model = model.to(device)
    
    key_list = list(CLASS_LABELS.keys())
    val_list = list(CLASS_LABELS.values())
    
    print(f"This script will predict a label given an URL. Enter 0 to quit.")
    
    while True:
        url_from_input = input("Please enter a URL you would like to check: ")
        if url_from_input == "0":
            exit()
        else:
            url = preprocess_text(url_from_input, max_length=MAX_URL_LENGTH)
            url = url.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
            model.eval()
            with torch.inference_mode():
                y_pred = model(url)
            
            print(f"URL: {convert_ascii_to_url(url)}")
            print(f"\tModel pred: \t{y_pred.argmax(dim=1).item()} - {key_list[val_list.index(y_pred.argmax(dim=1).item())]} with confidence { [np.round(i, 3) for i in torch.round(torch.softmax(y_pred, dim=1), decimals=3).tolist() ]}")
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