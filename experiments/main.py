import numpy as np
import pandas as pd 
from train_baseline_chichewa import * 
import argparse
import os

embedding_location = "../data/aligned_mt5_embeddings/"  
def results(df) -> None: 
    raise NotImplementedError

if __name__ == "__main__":
    # just original train data

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to load data from")
    parser.add_argument("embedding_type", type=str, help="file name of the generated embeddings")
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, "train.csv")
        
    train_and_evaluate(data_path)

    # original data + web scaraped data (no split)
    df = pd.read_csv("../data/train.csv")
    # df = df.append(pd.read_csv("../data/web_scraped_data/web_scraped_data.ny.csv"))
    # train_and_evaluate(data_path="", train_df=df)

    # original data + web scaraped data (3 sentence split)
    # df = df.append(pd.read_csv("../data/web_scraped_data/web_scraped_data_split_3.ny.csv"))
    # train_and_evaluate(data_path="", train_df=df)

    # original data + augmented data (every sentence split)
    # df = df.append(pd.read_csv("../data/augmented_data/augmented_data.ny.csv"))
    embedding_location = os.path.join(embedding_location, args.embedding_file)
    emb_orig = np.load(embedding_location)
    # emb_aug = np.load("../data/augmented_data/augmented_data.ny.npy")
    # emb = np.vstack([emb_orig, emb_aug])
    # train_and_evaluate(data_path="", train_df=df, train_emb=emb)

    # original data + augmented data (3 sentence split)
    # df = df.append(pd.read_csv("../data/augmented_data/augmented_data_split_3.ny.csv"))
    # emb_aug = np.load("../data/augmented_data/augmented_data.ny.npy")
    # emb = np.vstack([emb_orig, emb_aug])
    # train_and_evaluate(data_path="", train_df=df, train_emb=emb)