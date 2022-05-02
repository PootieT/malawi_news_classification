from typing import List, Dict, Optional, Tuple, Any


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from model_base import ClassificationModel
from mlp import NeuralNetwork, train
import xgboost as xgb
from read_data import get_data


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
        
        custom_embeddings = ["MTF", "Contrastive"]

        assert vectorizer in ["tfidf", "cv"] or vectorizer in custom_embeddings
        
        self.vectorizer_custom = False

        # print("this happened")
        # Vectorizer part

        if vectorizer not in custom_embeddings:
            self.vectorizer = {
                "tfidf": TfidfVectorizer,
                "cv": CountVectorizer,
            }[vectorizer]
            vectorizer_params = {} if vectorizer_params is None else vectorizer_params
            self.vectorizer = self.vectorizer(**vectorizer_params)

        else:
            self.vectorizer_custom = True
            self.custom_data, self.custom_labels = get_data(vectorizer)  
            self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
                         self.custom_data, self.custom_labels, test_size=0.2, random_state=42)

        assert classifier in ["NB", "LR", "RF", "XGB", "MLP"]
        

        # Classifier part
        self.classifier = {
            "NB": GaussianNB,
            "LR": LogisticRegression, 
            "MLP": NeuralNetwork,       # Need to add training loop
            "RF": RandomForestClassifier,
            # "XGB": xgb.XGBClassifier
        }[classifier]

        self.classifier_type = classifier
        model_params = {} if model_params is None else model_params
        self.classifier = self.classifier(**model_params)



    def train_supervised(self, train_data: List[str], train_labels: List[str]):

        if self.classifier_type not in ["MLP", "XGB"]:
            if ~self.vectorizer_custom:
                features = self.vectorizer.fit_transform(train_data).toarray()
                labels = [self.class2idx[c] for c in train_labels]
                self.classifier.fit(features, labels)
            else:
                features, labels = self.train_data, self.train_labels 
                labels = [self.class2idx[c] for c in train_labels]
                self.classifier.fit(features, labels)

        else:
            raise NotImplementedError


    def train_unsupervised(self, train_data: List[str]):
        raise NotImplementedError 


    def evaluate(self, test_data: List[str], test_labels: List[str]):
        """
        :param test_data: 
        :param test_labels:         
        """
        if self.classifier_type not in ["MLP", "XGB"]:

            if ~self.vectorizer_custom:
                features = self.vectorizer.transform(test_data).toarray()
                labels = [self.class2idx[c] for c in test_labels]
                pred = self.classifier.predict(features)
            else:
                features, labels = self.test_data, self.test_labels 
                labels = [self.class2idx[c] for c in labels]
                pred = self.classifier.predict(features, labels)
            
            metrics = super().get_metrics(labels, pred)
            return metrics

        else:
            raise NotImplementedError

