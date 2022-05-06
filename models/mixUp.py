import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sympy import GeneratorsNeeded
from tqdm import tqdm
import os 
import argparse
data_path = "../data/train.csv"



def generate_data(lambda_, c1_data, c2_data):
    generated_data = []
    count = 0
    for d1 in c1_data:
        for d2 in c2_data:
            if count == 2000:
                return generated_data
            mixedup_text = lambda_*d1 + (1 - lambda_)*d2 
            generated_data.append(mixedup_text)
            count += 1
    return generated_data

def mixup(text, classes, idx2class ,alpha=1.0):
    """
    Randomly mixes the given list of text embeddings with each other
    
    :param text: The text embeddings to be mixed up
    :param classes: 
    """
    unique_labels = np.unique(classes)    
    label_counts = {class_: classes.count(class_) for class_ in classes}

    created_embeddings = []
    created_labels = []

    generations  = []
    generations_labels = []

    generated_embeddings = dict()
    for class1 in tqdm(np.unique(classes), desc = "classes"):
        generated_embeddings[class1] = {
            "data":[],
            "labels":[],
            "data_len" : []
        }
        # data, labels = []
        for class2 in np.unique(classes):
            if class1 == class2:
                continue 
            # create at max max_count number of embeddings per combination
            max_count = max(classes.count(class1), classes.count(class2))
            if len(generated_embeddings[class1]["data"]) >= 2000:
                continue             
            c1_count = classes.count(class1)
            c2_count = classes.count(class2)

            r1, r2 = (c1_count / (c1_count + c2_count)), (c2_count / (c1_count + c2_count)) 
            
            c1_data = text[classes==class1]
            c2_data = text[classes==class2]

            if r1<r2:
                lambda_ = np.clip(np.random.beta(r2, r1), 0.6, 0.8)
                label_ = class1
                data = generate_data(lambda_ , c1_data, c2_data)
                generated_embeddings[class1]["data"] += data 
                generated_embeddings[class1]["labels"].append(label_)
                generations+=data 
                # generations_labels+=data
        
        print(len(generated_embeddings[class1]["data"])) 
        temp_classes = list(np.ones(len(generated_embeddings[class1]["data"]))*class1)
        generations_labels += temp_classes
        class__ = idx2class[class1]

    generations = np.array(generations)
    generated_data = np.vstack((text, generations))
    # generations_labels = [idx2class[c] for c in generations_labels]
    generations_labels = np.array(classes + generations_labels)
    print(generations_labels[:5])
    generations_labels = [idx2class[c] for c in generations_labels]
    print(generations_labels[:5])
    
    df = pd.DataFrame(generations_labels)
    with open('../data/mixup_data/final_embeddings.npy', 'wb') as f:
        np.save(f, generated_data)
        print(np.array(generated_data).shape)
    # with open('../data/mixup_data/final_labels.npy', 'wb') as f:
        # np.save(f, generations_labels)
    df.to_csv("../data/mixup_data/final_labels.csv")
    generated_embeddings[class1]["data"] = []
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to load data from")
    parser.add_argument("embeddings", type=str, help="file name of the generated embeddings")
    args = parser.parse_args()
        

    train_df = pd.read_csv(data_path)
    vectorizer = CountVectorizer(max_features=512)
    embeddings_path = os.path.join("../data/", args.embeddings + ".npy")


    text, labels = train_df.Text, list(train_df.Label)
    class2idx = {c: i for i, c in enumerate(labels)}
    idx2class = {i: c for i, c in enumerate(labels)}
    labels = [class2idx[c] for c in labels]

    # mtf_embeddings = np.load("../data/embeddings_chichewa.npy")
    mtf_small_embeddings = np.load(embeddings_path)
    text_embeddings = vectorizer.fit_transform(text).toarray()
    # mixedup_text, mixedup_labels = mixup(text, labels)
    mixup(mtf_small_embeddings,labels, idx2class)
