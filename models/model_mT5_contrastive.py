import logging
import math
from typing import List, Dict, Tuple, Callable, Iterable

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator, SentenceEvaluator
from torch import Tensor
from torch.utils.data import DataLoader

from models.model_base import ClassificationModel


logger = logging.getLogger(__name__)


class SupervisedLoss(nn.Module):
    """
    Supervised loss for classification on single sentence
    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()
    Example::
        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SupervisedLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])['sentence_embedding']

        output = self.classifier(rep)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return rep, output


class mT5Classifier(ClassificationModel):
    def __init__(
        self,
        supervised_ny: bool=True,
        supervised_en: bool=True,
        contrastive: bool=True,
        save_path: str="dump/mt5_classifier",
        verbose: bool=False
    ):
        super(mT5Classifier, self).__init__(verbose)
        base_model = models.Transformer("google/mt5-small")
        pooling_model = models.Pooling(base_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                                   activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[base_model, pooling_model, dense_model])
        self.train_loss_supervised = SupervisedLoss(model=self.model,
                                                    sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                                    num_labels=len(self.classes))
        self.classifier = self.train_loss_supervised.classifier

        self.supervised_ny = supervised_ny
        self.supervised_en = supervised_en
        self.contrastive = contrastive
        self.save_path = save_path

    def train_supervised_with_translation(
        self,
        train_data: List[str],
        train_labels: List[str],
        translated_data: List[str],
        **training_args
    ):
        assert len(train_data) == len(train_labels) == len(translated_data)
        assert hasattr(self, "evaluator"), "Call build_evaluator first to build an evaluator " \
                                           "for evaluation during training"

        supervised_train_samples = []
        if self.supervised_ny:
            for i in range(len(train_data)):
                supervised_train_samples.append(
                    InputExample(texts=[train_data[i]], label=self.class2idx[train_data[i]])
                )
        if self.supervised_en:
            for i in range(len(train_data)):
                supervised_train_samples.append(
                    InputExample(texts=[translated_data[i]], label=self.class2idx[train_data[i]])
                )
        train_dataloader_supervised = DataLoader(supervised_train_samples, shuffle=True, batch_size=training_args["batch_size"])

        if self.contrastive:
            contrastive_train_samples = []
            for i in range(len(train_data)):
                contrastive_train_samples.append(
                    InputExample(texts=[train_data[i], translated_data[i]]),
                )
            contrastive_dataloader = DataLoader(contrastive_train_samples, shuffle=True, batch_size=16)
            train_loss_contrastive = losses.MultipleNegativesSymmetricRankingLoss(model=self.model)

        num_epochs = 4

        warmup_steps = math.ceil(len(train_dataloader_supervised) * num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Here we define the two train objectives: train_dataloader_nli with train_loss_nli (i.e., SoftmaxLoss for NLI data)
        # and train_dataloader_sts with train_loss_sts (i.e., CosineSimilarityLoss for STSbenchmark data)
        # You can pass as many (dataloader, loss) tuples as you like. They are iterated in a round-robin way.
        train_objectives = [
            (train_dataloader_supervised, self.train_loss_supervised),
        ]
        if self.contrastive:
            train_objectives.append((contrastive_dataloader, train_loss_contrastive))

        # Train the model
        self.model.fit(
            train_objectives=train_objectives,
            evaluator=self.evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=self.save_path
       )

    def build_evaluator(self, test_data: List[str], test_labels: List[str]):
        test_data_samples = []
        for i in range(len(test_data)):
            test_data_samples.append(InputExample(texts=[test_data[i]], label=test_labels[i]))
        evaluator = LabelAccuracyEvaluator(self.model, "news_classification", self.classifier)
        self.evaluator = evaluator

    def evaluate(self, test_data: List[str], test_labels: List[str]) -> Dict:
        if not hasattr(self, "evaluator"):
            self.build_evaluator(test_data, test_labels)
        acc = self.evaluator(self.model, self.save_path)
        return {"accuracy": acc}
