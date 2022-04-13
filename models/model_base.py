from pprint import pprint
from typing import List, Dict
import pandas as pd
from sklearn.metrics import classification_report


class ClassificationModel:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.classes = sorted(['SOCIAL ISSUES', 'EDUCATION', 'RELATIONSHIPS', 'ECONOMY', 'RELIGION', 'POLITICS',
                               'LAW/ORDER', 'SOCIAL', 'HEALTH', 'ARTS AND CRAFTS', 'FARMING', 'CULTURE', 'FLOODING',
                               'WITCHCRAFT', 'MUSIC', 'TRANSPORT', 'WILDLIFE/ENVIRONMENT', 'LOCALCHIEFS', 'SPORTS',
                               'OPINION/ESSAY'])
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.idx2class = {i: c for i, c in enumerate(self.classes)}

    def train_supervised(self, train_data: List[str], train_labels: List[str]):
        pass

    def train_unsupervised(self, train_data: List[str]):
        pass

    def evaluate(self, test_data: List[str], test_labels: List[str]) -> Dict:
        pass

    def get_metrics(self, true_labels: List[str], pred_labels: List[str]):
        true_labels = [self.idx2class[c] for c in true_labels]
        pred_labels = [self.idx2class[c] for c in pred_labels]
        metrics = classification_report(true_labels, pred_labels, output_dict=True)
        if self.verbose:
            pprint(metrics)
        return metrics