import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '../models')
from model_baselines import BaselineClassifier

def train_and_evaluate(data_path: str):
    train_df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        train_df.Text, train_df.Label, test_size=0.3, random_state=42, stratify=train_df.Label)
    
    test_df = pd.read_csv("../data/test.csv")

    metrics = []
    combo = []
    for vec in ["tfidf", "cv", "MT5", "CONTRASTIVE"]:
        for m in ["NB", "LR", "XGB", "MLP"]:
            combo.append(vec + "-" + m)
            model = BaselineClassifier(vec, m, verbose=False)
            model.train_supervised(X_train, y_train)
            metric = model.evaluate(X_test, y_test)
            metric["vectorizer"], metric["classifier"] = vec, m
            metrics.append(metric)

    eval_df = pd.DataFrame(metrics)
    eval_df.index = combo
    return eval_df




