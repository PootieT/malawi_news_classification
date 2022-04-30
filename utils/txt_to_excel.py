import re
import os
import json
from tqdm import tqdm
import pandas as pd
import datasets


def load_jsonl_to_pd(path: str):
    dataset = datasets.load_dataset('text', data_files={'train': [path]})["train"]
    dataset = dataset.map(lambda example: {"content": json.loads(example["text"])["text"]}, remove_columns=["text"])
    dataset = dataset.rename_column("content", "text")
    return pd.DataFrame(dataset["text"], columns=["text"])


# def convert_dir(path: str):
#     for f in tqdm(os.listdir(path)):
#         f_path = f"{path}/{f}"
#         if not os.path.exists(f_path.replace(".txt", ".xlsx")):
#             convert_txt_to_excel(f_path)


def convert_jsonl_to_excel(path: str, max_lines: int=1500):
    df = load_jsonl_to_pd(path)
    # df.text = df.text.apply(lambda x: re.sub(r'@@\d*', '', x).strip())
    post_fix = path.split(".")[-1]
    idx = post_fix.replace("jsonl", "")
    dir_path = path.replace(f".{post_fix}", f"_{idx}")
    os.makedirs(dir_path, exist_ok=True)
    for idx, i in tqdm(enumerate(range(0, len(df), max_lines)), total=len(df)//max_lines):
        df.iloc[i:i+max_lines].to_excel(f"{dir_path}/{i}.xlsx", index=False, header=False)


def aggregate_excels(path: str):
    df = pd.DataFrame()
    ordered_dir = sorted(os.listdir(path), key=lambda x: int(x.split(".")[0]))
    for f in tqdm(ordered_dir):
        f_path = f"{path}/{f}"
        df_s = pd.read_excel(f_path, header=None, names=["text"])
        df = df.append(df_s)
    df.to_excel(f"{path}/0_45000.xlsx", index=False, header=False)


def get_english(path: str, max_lines: int=1500):
    df = load_jsonl_to_pd(path)
    # df.text = df.text.apply(lambda x: re.sub(r'@@\d*', '', x).strip())
    post_fix = path.split(".")[-1]
    idx = post_fix.replace("jsonl", "")
    dir_path = path.replace(f".{post_fix}", f"_{idx}")
    os.makedirs(dir_path, exist_ok=True)
    df_new = pd.DataFrame()
    for idx, i in tqdm(enumerate(range(0, 45000, max_lines)), total=30):
        df_new = df_new.append(df.iloc[i:i+max_lines])
    df_new.to_excel(f"0_45000.xlsx", index=False, header=False)


if __name__ == "__main__":
    # convert_jsonl_to_excel("../data/english_news/realnews/realnews.jsonl69")
    # convert_txt_to_excel("../data/english_news/coca-samples-text/text_acad.txt")
    # convert_dir("/home/zilu/projects/cs_505/malawi_news_classification/data/english_news/coca-samples-text")
    aggregate_excels("../data/english_news/realnews/realnews_69_ny")
    # get_english("../data/english_news/realnews/realnews.jsonl69")