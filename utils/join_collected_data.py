import os.path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from utils import expand_sentences


def join_collected_data(path: str, sents_per_split: Optional[int]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    assume data is in folder provided path, collected english news are
    named with their class label, and chichewa translations are named
    {class}_chichewa.csv
    :param path: path to folder containing additional news data collected by Keanu
    :param sents_per_split: if splitting data by sentences, it is the number of sentences
        per split used in splitting. If None, no splitting happens
    :return : Chichewa and English news with Text and Label columns (also saved)
        if split sentence is used, return has additional column of which news article the
        sentence come from
    """
    if not os.path.exists(path):
        raise Exception("Data path not found, please download 'web_scraped_data' from google"
                        "drive and pass in the path and re-execute.")
    classes = {"FLOODING": "flood",
               "MUSIC": "music",
               "WITCHCRAFT": "witch",
               "WILDLIFE/ENVIRONMENT": "wildlife",
               "HEALTH": "health"}
    df_en, df_ny = pd.DataFrame(), pd.DataFrame()
    for class_name, file_str in classes.items():
        df_en_class = pd.read_csv(f"{path}/{file_str}.csv")
        df_ny_class = pd.read_csv(f"{path}/{file_str}_chichewa.csv")
        df_en_class["Label"] = class_name
        df_en = df_en.append(df_en_class)
        df_ny = df_ny.append(df_ny_class)

    df_ny.to_csv(f"{path}/web_scraped_data.ny.csv", index=False)
    df_en.to_csv(f"{path}/web_scraped_data.en.csv", index=False)
    if sents_per_split is not None:
        ny_text, en_text, y, news_idx = expand_sentences(df_ny.Text.tolist(), df_en.Text.tolist(), df_ny.Label.tolist(), sents_per_split=sents_per_split)
        df_ny = pd.DataFrame({"Text": ny_text, "Label": y, "Index": news_idx})
        df_en = pd.DataFrame({"Text": en_text, "Label": y, "Index": news_idx})
        df_ny = df_ny[df_ny.Text.str.len() > 1]
        df_en = df_en[df_en.Text.str.len() > 1]
        df_ny.to_csv(f"{path}/web_scraped_data_split_{sents_per_split}.ny.csv")
        df_en.to_csv(f"{path}/web_scraped_data_split_{sents_per_split}.en.csv")
    return df_ny, df_en


if __name__ == "__main__":
    join_collected_data("../data/web_scraped_data", sents_per_split=3)