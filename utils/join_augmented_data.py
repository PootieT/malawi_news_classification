import os
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import numpy as np
import pandas as pd


def condense_sentences(df: pd.DataFrame, repeat_per_class: Dict[str, int], sents_per_split: int=3) -> pd.DataFrame:
    """
    Given a pandas df with Text and Label column, where each row of text is a single sentence,
    we combine them every few sentences to provide a more condensed dataset.
    """
    condense_df = []
    for cls in tqdm(df.Label.unique()):
        df_cls = df[df.Label == cls]
        no_repeat_indices = np.arange(0, len(df_cls), repeat_per_class[cls])
        for offset_idx in np.arange(0, len(no_repeat_indices), sents_per_split):
            for repeat_idx in range(repeat_per_class[cls]):
                df_small = df_cls.iloc[no_repeat_indices[offset_idx:offset_idx+sents_per_split]+repeat_idx]
                condense_df.append({"Text": ". ".join(df_small.Text.tolist()), "Label": cls})
    condense_df = pd.DataFrame(condense_df)
    return condense_df


def join_augmented_data(path: str, sents_per_split: Optional[int]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    assume data is in folder provided path (augmented_data), with 'Train_Augmented_2.csv' file there
    :param path: path to folder containing additional news data augmented by Sherry
    :param sents_per_split: if splitting data by sentences, it is the number of sentences
        per split used in splitting. If None, no splitting happens
    :return : Chichewa and English news with Text and Label columns (also saved)
        if split sentence is used, return has additional column of which news article the
        sentence come from
    """
    if not os.path.exists(path):
        raise Exception("Data path not found, please download 'augmented_data' folder from google"
                        "drive and pass in the path and re-execute.")
    # number of augmented sentence per original sentence generated
    class2cnt = {
        "ECONOMY": 4, "WITCHCRAFT": 20, "RELATIONSHIPS": 5,
        "FARMING": 4, "MUSIC": 13, "WILDLIFE/ENVIRONMENT": 13,
        "SPORTS":7, "OPINION/ESSAY":10, "EDUCATION":11,
        "LOCALCHIEFS":10, "CULTURE": 9, "ARTS AND CRAFTS": 40,
        "FLOODING": 40,
    }
    df = pd.read_csv(f"{path}/Train_Augmented_2.csv")
    df_ny = df[["chichewa translation", "label"]]
    df_ny = df_ny.rename(columns={"chichewa translation": "Text", "label": "Label"})
    df_en = df[["english augmentations", "label"]]
    df_en = df_en.rename(columns={"english augmentations": "Text", "label": "Label"})

    df_ny.to_csv(f"{path}/augmented_data.ny.csv", index=False)
    df_en.to_csv(f"{path}/augmented_data.en.csv", index=False)
    if sents_per_split is not None:
        df_ny = condense_sentences(df_ny, class2cnt, sents_per_split)
        df_en = condense_sentences(df_en, class2cnt, sents_per_split)
        df_ny.to_csv(f"{path}/augmented_data_split_{sents_per_split}.ny.csv", index=False)
        df_en.to_csv(f"{path}/augmented_data_split_{sents_per_split}.en.csv", index=False)

    return df_ny, df_en


if __name__ == "__main__":
    join_augmented_data("../data/augmented_data", sents_per_split=3)
