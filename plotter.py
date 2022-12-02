
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from train_url_detector import URLDataset
import pickle
import torch


def main():
    """Extracts the content and results of model training from the pickle file and plots the training and test loss curves"""
    file = open("./training_hist/train_hist.pickle", "rb")
    
    data_dict = pickle.load(file)
    file.close()
    
    train_loss = data_dict["train_loss_hist"]
    train_acc = data_dict["train_acc_hist"]
    test_loss = data_dict["test_loss_hist"]
    test_acc = data_dict["test_acc_hist"]
    
    train_loss = torch.tensor(train_loss, device = 'cpu')
    train_acc = torch.tensor(train_acc, device = 'cpu')
    test_loss = torch.tensor(test_loss, device = 'cpu')
    test_acc = torch.tensor(test_acc, device = 'cpu')
    # train_loss, train_acc = train_loss.to("cpu"), train_acc.to("cpu")
    # test_loss, test_acc = test_loss.to("cpu"), test_acc.to("cpu")
    
    num_epochs = range(len(train_loss))
    
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(num_epochs, train_loss, label="train_loss")
    plt.plot(num_epochs, test_loss, label="test_loss")
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(num_epochs, train_acc, label="train_accuracy")
    plt.plot(num_epochs, test_acc, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()   
    
    plt.show()
    
if __name__ == "__main__":
    main()