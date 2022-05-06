import numpy as np
import pandas as pd 
from train_baseline_chichewa import * 


def results(df) -> None: 
    raise NotImplementedError

if __name__ == "__main__":
    # just original train data
    train_and_evaluate(data_path="../data/train.csv")

    # original data + web scaraped data (no split)
    df = pd.read_csv("../data/train.csv")
    # df = df.append(pd.read_csv("../data/web_scraped_data/web_scraped_data.ny.csv"))
    # train_and_evaluate(data_path="", train_df=df)

    # original data + web scaraped data (3 sentence split)
    # df = df.append(pd.read_csv("../data/web_scraped_data/web_scraped_data_split_3.ny.csv"))
    # train_and_evaluate(data_path="", train_df=df)

    # original data + augmented data (every sentence split)
    # df = df.append(pd.read_csv("../data/augmented_data/augmented_data.ny.csv"))
    emb_orig = np.load("../data/aligned_mt5_embeddings/embeddings_chichewa.npy")
    # emb_aug = np.load("../data/augmented_data/augmented_data.ny.npy")
    # emb = np.vstack([emb_orig, emb_aug])
    # train_and_evaluate(data_path="", train_df=df, train_emb=emb)

    # original data + augmented data (3 sentence split)
    # df = df.append(pd.read_csv("../data/augmented_data/augmented_data_split_3.ny.csv"))
    # emb_aug = np.load("../data/augmented_data/augmented_data.ny.npy")
    # emb = np.vstack([emb_orig, emb_aug])
    # train_and_evaluate(data_path="", train_df=df, train_emb=emb)