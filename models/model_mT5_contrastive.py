import os
import csv
import json
import logging
import math
from typing import List, Dict, Tuple, Callable, Iterable, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator, SentenceEvaluator, SimilarityFunction
from sentence_transformers.util import batch_to_device
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import MT5EncoderModel, AutoConfig

from model_base import ClassificationModel


logger = logging.getLogger(__name__)


class ModifiedLabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        name: str = "",
        softmax_model = None,
        write_csv: bool = True,
        compute_metrics: Optional[Callable]=None
    ):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.compute_metrics = compute_metrics

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        pred_labels = torch.tensor([])

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            pred_labels_batch = torch.argmax(prediction, dim=1)
            correct += pred_labels_batch.eq(label_ids).sum().item()
            pred_labels = torch.cat([pred_labels, pred_labels_batch.cpu()])
        accuracy = correct/total

        # logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        print("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        if self.compute_metrics is not None:
            true_labels = [d.label for d in self.dataloader.dataset]
            metrics = self.compute_metrics(true_labels, pred_labels.numpy())
        else:
            metrics = None

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])
            if metrics is not None:
                with open(f"{os.path.dirname(csv_path)}/classification_report_{epoch}_{step}.json", "w") as f:
                    json.dump(metrics, f)

        return accuracy


class ContrastiveLossEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its softmax on a labeled dataset
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, sentences1: List[str], sentences2: List[str], batch_size: int = 16,
                 main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False,
                 write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                        logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson",
                            "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson",
                            "dot_spearman"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []

        for example in examples:
            # a very rough optimistic truncation that prevents super eval sequences from jamming up GPU mem
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
        return cls(sentences1, sentences2, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        cosine_scores = np.mean(1 - (paired_cosine_distances(embeddings1, embeddings2))).item()

        logger.info("Cosine-Score :".format(cosine_scores))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, cosine_scores])

        return cosine_scores


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
        # modified forward so it works with evaluator call
        # self.classifier = LinearClassifier(sentence_embedding_dimension, num_labels)
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
        supervised_en: bool=False,
        contrastive: bool=False,
        save_path: str="dump/mt5_classifier",
        training_args: Optional[Dict[str, Any]]=None,
        verbose: bool=False,
        load_model_path: Optional[str]=None,
        load_dense_layer: bool=True
    ):
        super(mT5Classifier, self).__init__(verbose)
        model_name = "google/mt5-small"
        base_model = models.Transformer(model_name)
        # sentence bert repo does not current support mT5 yet (only T5 ironically)
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        mT5_config = AutoConfig.from_pretrained(model_name)
        base_model.auto_model = MT5EncoderModel.from_pretrained("google/mt5-small", config=mT5_config)

        pooling_model = models.Pooling(base_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                                   activation_function=nn.Tanh())

        if load_model_path is not None:
            base_model.auto_model.load_state_dict(torch.load(f"{load_model_path}/pytorch_model.bin", map_location=torch.device('cpu')))
            if load_dense_layer:
                dense_model.load_state_dict(torch.load(f"{load_model_path}/2_Dense/pytorch_model.bin", map_location=torch.device('cpu')))

        if load_dense_layer:
            self.model = SentenceTransformer(modules=[base_model, pooling_model, dense_model])
        else:
            self.model = SentenceTransformer(modules=[base_model, pooling_model])
        self.model.max_seq_length = 512
        self.train_loss_supervised = SupervisedLoss(model=self.model,
                                                    sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                                    num_labels=len(self.classes))
        # self.classifier = self.train_loss_supervised.classifier

        self.supervised_ny = supervised_ny
        self.supervised_en = supervised_en
        self.contrastive = contrastive
        self.save_path = save_path

        default_training_args = {"batch_size": 16, "epochs": 4, "evaluation_steps": 1000}
        if training_args is None:
            training_args = default_training_args
        else:
            for k, v in default_training_args.items():
                if k not in training_args:
                    training_args[k] = default_training_args[k]
        self.training_args = training_args

    def trim_long_sentences(self, sents: List[str], max_len: int=512):
        """
        optimistic way of trimming down long tail sentences
        :param sents:
        :param max_len:
        :return:
        """
        return [" ".join(s.split()[:max_len]) for s in sents]

    def pretrain(self, data: List[str], translated_data: List[str]):
        data = self.trim_long_sentences(data)
        translated_data = self.trim_long_sentences(translated_data)

        contrastive_train_samples = []
        for i in range(len(data)):
            contrastive_train_samples.append(
                InputExample(texts=[data[i], translated_data[i]]),
            )
        contrastive_dataloader = DataLoader(contrastive_train_samples, shuffle=True,
                                            batch_size=self.training_args["batch_size"])
        train_loss_contrastive = losses.MultipleNegativesSymmetricRankingLoss(model=self.model)

        num_epochs = self.training_args["epochs"]

        warmup_steps = math.ceil(len(contrastive_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        train_objectives = [(contrastive_dataloader, train_loss_contrastive)]

        # Train the model
        self.model.fit(
            train_objectives=train_objectives,
            evaluator=self.evaluator,
            epochs=num_epochs,
            evaluation_steps=self.training_args["evaluation_steps"],
            warmup_steps=warmup_steps,
            output_path=self.save_path,
            use_amp=False,
            # mT5 model with amp training results in nan https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
            optimizer_params={'lr': 3e-6}
        )

    def train_supervised_with_translation(
        self,
        train_data: List[str],
        train_labels: List[str],
        translated_data: Optional[List[str]],
    ):
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_multi-task.py
        assert len(train_data) == len(train_labels)
        assert hasattr(self, "evaluator"), "Call build_evaluator first to build an evaluator " \
                                           "for evaluation during training"
        assert translated_data is not None or (not self.supervised_en and not self.contrastive)
        supervised_train_samples = []
        if self.supervised_ny:
            for i in range(len(train_data)):
                supervised_train_samples.append(
                    InputExample(texts=[train_data[i]], label=self.class2idx[train_labels[i]])
                )
        if self.supervised_en:
            for i in range(len(train_data)):
                supervised_train_samples.append(
                    InputExample(texts=[translated_data[i]], label=self.class2idx[train_labels[i]])
                )
        train_dataloader_supervised = DataLoader(supervised_train_samples, shuffle=True, batch_size=self.training_args["batch_size"])

        if self.contrastive:
            contrastive_train_samples = []
            for i in range(len(train_data)):
                contrastive_train_samples.append(
                    InputExample(texts=[train_data[i], translated_data[i]]),
                )
            contrastive_dataloader = DataLoader(contrastive_train_samples, shuffle=True, batch_size=self.training_args["batch_size"])
            train_loss_contrastive = losses.MultipleNegativesSymmetricRankingLoss(model=self.model)

        num_epochs = self.training_args["epochs"]

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
            evaluation_steps=self.training_args["evaluation_steps"],
            warmup_steps=warmup_steps,
            output_path=self.save_path,
            use_amp=False,  # mT5 model with amp training results in nan https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
            optimizer_params={'lr': 3e-6}
       )

    def build_evaluator(self, test_data: List[str], test_labels: List[str]):
        test_data_samples = []
        for i in range(len(test_data)):
            test_data_samples.append(InputExample(texts=[test_data[i]], label=self.class2idx[test_labels[i]]))
        eval_dataloader = DataLoader(test_data_samples, shuffle=True, batch_size=self.training_args["batch_size"])
        evaluator = ModifiedLabelAccuracyEvaluator(eval_dataloader, "news_classification", self.train_loss_supervised, compute_metrics=super().get_metrics)
        self.evaluator = evaluator

    def build_pretrain_evaluator(self, test_data: List[str], translated_data: List[str]):
        test_data_samples = []
        test_data = self.trim_long_sentences(test_data)
        translated_data = self.trim_long_sentences(translated_data)
        for i in range(len(test_data)):
            test_data_samples.append(InputExample(texts=[test_data[i], translated_data[i]]))
        # eval_dataloader = DataLoader(test_data_samples, shuffle=True, batch_size=self.training_args["batch_size"])
        evaluator = ContrastiveLossEvaluator.from_input_examples(test_data_samples,
                                                                 batch_size=self.training_args["batch_size"],
                                                                 main_similarity=SimilarityFunction.COSINE,
                                                                 name="eval_similarity")
        self.evaluator = evaluator

    def evaluate(self, test_data: List[str], test_labels: List[str]) -> Dict:
        if not hasattr(self, "evaluator"):
            self.build_evaluator(test_data, test_labels)
        acc = self.evaluator(self.model, self.save_path)
        return {"accuracy": acc}
