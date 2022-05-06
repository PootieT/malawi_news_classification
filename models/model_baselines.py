from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from model_base import ClassificationModel
from mlp import NeuralNetwork
import xgboost as xgb
from read_data import get_data

custom_embeddings = ["MT5", "Contrastive"]

class BaselineClassifier(ClassificationModel):
    """
    baseline classification pipeline that tokenizes text using basic vectorizer
    and classify using sklearn shallow models
    """
    def __init__(
        self,
        vectorizer: str,
        classifier: str,
        vectorizer_params: Dict[str, Any]=None,
        model_params: Dict[str, Any]=None,
        verbose: bool=True
    ):
        super(BaselineClassifier, self).__init__(verbose)

        assert vectorizer in ["tfidf", "cv", "MT5", "Contrastive"]

        self.vectorizer_custom = False

        if vectorizer not in custom_embeddings:
            self.vectorizer = {
                "tfidf": TfidfVectorizer,
                "cv": CountVectorizer,
            }[vectorizer]
            vectorizer_params = {} if vectorizer_params is None else vectorizer_params
            self.vectorizer = self.vectorizer(**vectorizer_params)

        else:
            print("this happened")
            self.vectorizer_custom = True
            self.custom_vectorizer_type = vectorizer

        assert classifier in ["NB", "LR", "RF", "XGB", "MLP"]


        # Classifier part
        self.classifier = {
            "NB": GaussianNB,
            "LR": LogisticRegression,
            "MLP": NeuralNetwork,       # Need to add training loop
            "RF": RandomForestClassifier,
            "XGB": xgb.XGBClassifier
        }[classifier]

        self.classifier_type = classifier
        model_params = {} if model_params is None else model_params
        self.classifier = self.classifier(**model_params)



    def train_supervised(
        self,
        train_data: List[str],
        train_labels: List[str],
        train_emb: Optional[np.array]
    ):
        if self.vectorizer_custom == False: # for tfidf, cv
            features = self.vectorizer.fit_transform(train_data).toarray()

        else: # for MT5, contrastive
            # embeddings = get_data(self.custom_vectorizer_type)
            features = train_emb[train_data.index]

        labels = [self.class2idx[c] for c in train_labels]
        self.classifier.fit(features, labels)


    def evaluate(self, test_data: List[str], test_labels: List[str], train_emb: Optional[np.array]):
        """
        :param test_data:
        :param test_labels:
        """

        if self.vectorizer_custom == False:
            features = self.vectorizer.transform(test_data).toarray()
        else:
            # embeddings = get_data(self.custom_vectorizer_type)
            features = train_emb[test_data.index]

        labels = [self.class2idx[c] for c in test_labels]
        pred = self.classifier.predict(features)
        metrics = super().get_metrics(labels, pred)
        return metrics

    def predict(self, test_data: List[str], test_emb: Optional[np.array]=None):
        if self.vectorizer_custom == False:
            features = self.vectorizer.transform(test_data).toarray()
        else:
            features = test_emb
        pred = self.classifier.predict(features)
        return [self.idx2class[p] for p in pred]
