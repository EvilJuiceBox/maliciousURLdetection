
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from train_url_detector import URLDataset
import pickle
import torch
import os

from train_url_detector import URL_Neural_Net

from torchmetrics import Accuracy, ConfusionMatrix

from mlxtend.plotting import plot_confusion_matrix 



def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MAX_URL_LENGTH = 2175
    """Hardcoded class labels, may have to change this per new model"""
    CLASS_LABELS = {'defacement': 0, 'benign': 1, 'phishing': 2, 'malware': 3}
    
    model = URL_Neural_Net(input_shape=1, hidden_units=256, output_shape=len(CLASS_LABELS))
    model.load_state_dict(torch.load("./models/model.pth"))
    model = model.to(device)
    
    
    file = open("./training_hist/train_hist.pickle", "rb")
    
    data_dict = pickle.load(file)
    file.close()
    
    test_set = data_dict["test_data"]

    # urls, labels = test_set[:]
    
    
    # # ## load the data into dataloader
    BATCH_SIZE = 32
    NUM_WORKER = min(1, os.cpu_count()-2)
    test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)

    y_test = []
    y_test_pred = []
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        model.eval()
        with torch.inference_mode():
            X = X.unsqueeze(dim=1)
            y_logits = model(X)
            y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
            y_test.extend(y)
            y_test_pred.extend(y_preds)
        
    
    y_test = torch.tensor(y_test, dtype=torch.int64).to(device)
    y_test_pred = torch.tensor(y_test_pred, dtype=torch.int64).to(device)
    
    tm_accuracy = Accuracy().to(device)
    print(f"The accuracy of the model is: {tm_accuracy(y_test_pred, y_test).item()}")
    
    tm_confusion = ConfusionMatrix(num_classes=4).to(device)
    conf_matrix = tm_confusion(y_test_pred, y_test).cpu().numpy()
    print(f"The Confusion Matrix of the model is: \n{conf_matrix}")
    print("")
    
    print(f"debug sum {np.sum(conf_matrix, axis=1)}")
    percentage = conf_matrix / np.sum(conf_matrix, axis=1)[:, None]
    percentage = np.round_(percentage*100, decimals=2)
    print(f"The Confusion Matrix of the model as percentages is: \n{percentage}")
    
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_matrix,
        class_names=CLASS_LABELS,
        figsize=(10,7),
        show_absolute=True,
        show_normed=True,
        colorbar=True
    )
    plt.show()
    
    # fig, ax = plot_confusion_matrix(
    # conf_mat=percentage,
    # class_names=CLASS_LABELS,
    # figsize=(10,7)
    # )
    # plt.show()
    

    
if __name__ == "__main__":
    main()