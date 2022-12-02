Course semester for CSE891, developed by Kira Chan (Department of Computer Science and Engineering, Michigan State University, East Lansing)
Contact: chanken1@msu.edu

This project uses a DNN to predict URL inputs and determines if they are benign, phishing, defacement, or malware. The full dataset is obtained from Kaggle:
	https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset by Manu Siddhartha.

Model accuracy:
	Train: 0.94 - Test: 0.94

Please see \sample outputs\ folder for some analysis result or sample outputs.


Instructions to run:

1) Install pytorch, cuda support from https://pytorch.org/get-started/locally/ and https://developer.nvidia.com/cuda-11-7-0-download-archive
2) Create a venv using python -m venv venv
3) Activate the venv (venv\Scripts\activate on windows)
4) Install requirements using pip install -r requirements.txt

############### A pretrained model is availabe in model/model.pth  ################

Instructions to train:

1) The training script is **train_url_detector.py**
2) The command to run the training script is simply python train_url_detector.py
	Notes:
	a) The dataset is rather large, thus analysis and training will take a long time if you do not use GPU. The number of samples can be reduced in the class URLDataset
		by uncommenting the line that divides the dataset.
	b) Please uncomment the sorted(...) line in URL_Neural_Net init function to obtain a consistent label in the future.


Instructions to run analysis scripts:
	Plotting: python plotter.py
	Testing with random data from dataset: python predict_url.py
	Testing by category (set the number of samples): predict_by_category.py
	Testing with custom URL for entry: python predict_custom_url.py, enter the URL when promoted. Enter 0 to quit.
	Confusion matrix: confusionmatrix.py

