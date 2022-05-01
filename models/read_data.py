import numpy as np
import pandas as pd 
import pickle 
import os 

PATH = "../data/custom_data/"

MT5 = "mt5_embeddings"
CONTASTIVE = "contrastive_embeddings"


def read_data(file):
    
    extension = os.path.splitext(file)[1]

    if extension == "npz":
        file = np.load(file)
        data = file["arr_0"]
        labels = file["arr_1"]
        return data, labels 

    if extension == "csv":
        file = pd.read_csv(file)
        data = file.drop(["labels"], axis=1).to_numpy()
        labels = list(file["labels"])
        return data, labels

def get_data(vectorizer):

    if vectorizer == "MT5":
        return read_data(MT5)
    
    elif vectorizer == "CONTRASTIVE":
        return read_data(CONTASTIVE)



    