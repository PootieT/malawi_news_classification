import re
from typing import Optional, List

import json
import datasets
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from models.model_mT5_contrastive import mT5Classifier
from utils.utils import expand_sentences


def train_and_evaluate(data_path: str, translated_data_path: str):
    train_df = pd.read_csv(data_path)
    train_t_df = pd.read_csv(translated_data_path)
    assert len(train_t_df) == len(train_df)

    X_train, X_test, y_train, y_test, X_t_train, X_t_test = train_test_split(
        train_df.Text, train_df.Label, train_t_df.Text, test_size=0.3, random_state=42, stratify=train_df.Label)

    X_train, X_t_train, y_train = expand_sentences(X_train.tolist(), X_t_train.tolist(), y_train.tolist())
    X_test, X_t_testn, y_test = expand_sentences(X_test.tolist(), X_t_test.tolist(), y_test.tolist())

    # test_df = pd.read_csv("../data/test.csv")

    metrics = []
    training_args = {"batch_size": 8, "epochs": 2, "evaluation_steps": 200}
    model = mT5Classifier(
        supervised_ny=True,
        supervised_en=True,
        contrastive=True,
        save_path="dump/mt5_sup_chi",
        training_args=training_args,
        verbose=True
    )
    model.build_evaluator(X_test, y_test)
    model.train_supervised_with_translation(X_train, y_train, X_t_train)
    metric = model.evaluate(X_test, y_test)
    metric["supervised_ny"] = True
    metric["supervised_en"] = False
    metric["contrastive"] = False
    metrics.append(metric)

    eval_df = pd.DataFrame(metrics)
    return eval_df


def load_jsonl_to_pd(path: str):
    dataset = datasets.load_dataset('text', data_files={'train': [path]})["train"]
    dataset = dataset.map(lambda example: {"content": json.loads(example["text"])["text"]}, remove_columns=["text"])
    dataset = dataset.rename_column("content", "text")
    return pd.DataFrame(dataset["text"], columns=["text"])


def read_txt_to_pandas(path: str):
    # only this way do we preserve the way original translation newline
    # was appended to keep the corpus parallel
    with open(path, "r") as f:
        lines = [s.strip() for s in f.readlines()]
    return pd.DataFrame(lines, columns=["text"])


def pretrain_mt5(data_path: str, translated_data_path: str):
    if ".jsonl" in data_path:
        train_df = load_jsonl_to_pd(data_path)
        train_t_df = read_txt_to_pandas(translated_data_path)
        # train_t_df = pd.read_table(translated_data_path, names=["text"],sep="\n")
    elif ".xlsx" in data_path:
        train_df = pd.read_excel(data_path, header=None, names=["text"])
        train_t_df = pd.read_excel(translated_data_path, header=None, names=["text"])
    else:
        train_df = pd.read_csv(data_path)
        train_t_df = pd.read_csv(translated_data_path)
    if len(train_df) > len(train_t_df):
        train_df = train_df[: len(train_t_df)]
    elif len(train_t_df) > len(train_df):
        train_t_df = train_t_df[: len(train_df)]

    assert len(train_t_df) == len(train_df)

    X_train, X_test, X_t_train, X_t_test = train_test_split(
        train_df.text, train_t_df.text, test_size=0.005, random_state=42)

    if ".jsonl" in data_path:
        X_train, X_t_train = expand_sentences(X_train.tolist(), X_t_train.tolist(), method="s2s")
        X_test, X_t_test = expand_sentences(X_test.tolist(), X_t_test.tolist(), method="s2s")
    elif ".xlsx" in data_path:
        X_train, X_t_train = expand_sentences(X_train.tolist(), X_t_train.tolist(), method="interpolate")
        X_test, X_t_test = expand_sentences(X_test.tolist(), X_t_test.tolist(), method="interpolate")

    # test_df = pd.read_csv("../data/test.csv")

    metrics = []
    training_args = {"batch_size": 4, "epochs": 2, "evaluation_steps": 200}
    model = mT5Classifier(
        supervised_ny=True,
        supervised_en=True,
        contrastive=True,
        save_path="dump/mt5_sup_chi",
        training_args=training_args,
        verbose=True
    )
    model.build_pretrain_evaluator(X_test, X_t_test)
    model.pretrain(X_train, X_t_train)
    metric = model.evaluate(X_test, X_t_test)
    metrics.append(metric)

    eval_df = pd.DataFrame(metrics)
    return eval_df


if __name__ == "__main__":
    # pretrain_mt5("../data/english_news/realnews/realnews.jsonl00",
    #              "../data/english_news/realnews/realnews_ny.jsonl00")

    pretrain_mt5("../data/english_news/realnews/realnews_69/0_45000_en.xlsx",
                 "../data/english_news/realnews/realnews_69_ny/0_45000_ny.xlsx")
    # chichewa_result = train_and_evaluate(data_path="../data/train.csv",
    #                                      translated_data_path="../data/train_google_translated.csv")
    pass