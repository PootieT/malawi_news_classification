import pandas as pd
from sklearn.model_selection import train_test_split

from models.model_baselines import BaselineClassifier


def train_and_evaluate(data_path: str):
    train_df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        train_df.Text, train_df.Label, test_size=0.3, random_state=42, stratify=train_df.Label)

    # test_df = pd.read_csv("../data/test.csv")

    metrics = []
    for vec in ["tfidf", "cv"]:
        for m in ["NB", "LR"]:
            model = BaselineClassifier(vec, m, verbose=False)
            model.train_supervised(X_train, y_train)
            metric = model.evaluate(X_test, y_test)
            metric["vectorizer"], metric["classifier"] = vec, m
            metrics.append(metric)

    eval_df = pd.DataFrame(metrics)
    return eval_df


if __name__ == "__main__":
    chichewa_result = train_and_evaluate(data_path="../data/train.csv")
    english_result = train_and_evaluate(data_path="../data/train_google_translated.csv")
    pass