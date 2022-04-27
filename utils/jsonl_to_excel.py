import re
import os
from tqdm import tqdm
import pandas as pd


def convert_dir(path:str):
    for f in tqdm(os.listdir(path)):
        f_path = f"{path}/{f}"
        if not os.path.exists(f_path.replace(".txt", ".xlsx")):
            convert_txt_to_excel(f_path)


def convert_txt_to_excel(path: str):
    df = pd.read_table(path, names=["text"])
    df.text = df.text.apply(lambda x: re.sub(r'@@\d*', '', x).strip())
    df.to_excel(path.replace(".txt", ".xlsx"))


if __name__ == "__main__":

    # convert_txt_to_excel("../data/english_news/coca-samples-text/text_acad.txt")
    convert_dir("/home/zilu/projects/cs_505/malawi_news_classification/data/english_news/coca-samples-text")