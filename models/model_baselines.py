from typing import List, Dict, Optional, Tuple, Any


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from model_base import ClassificationModel


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
        assert vectorizer in ["tfidf", "cv"]
        self.vectorizer = {
            "tfidf": TfidfVectorizer,
            "cv": CountVectorizer
        }[vectorizer]
        vectorizer_params = {} if vectorizer_params is None else vectorizer_params
        self.vectorizer = self.vectorizer(**vectorizer_params)

        assert classifier in ["NB", "LR"]
        self.classifier = {
            "NB": GaussianNB,
            "LR": LogisticRegression
        }[classifier]
        model_params = {} if model_params is None else model_params
        self.classifier = self.classifier(**model_params)

    def train_supervised(self, train_data: List[str], train_labels: List[str]):
        features = self.vectorizer.fit_transform(train_data).toarray()
        labels = [self.class2idx[c] for c in train_labels]
        self.classifier.fit(features, labels)

    def train_unsupervised(self, train_data: List[str]):
        raise NotImplementedError

    def evaluate(self, test_data: List[str], test_labels: List[str]):
        features = self.vectorizer.transform(test_data).toarray()
        labels = [self.class2idx[c] for c in test_labels]
        pred = self.classifier.predict(features)
        metrics = super().get_metrics(labels, pred)
        return metrics

