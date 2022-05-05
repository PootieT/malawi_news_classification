import json
import os
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
import datasets

import torch
from transformers import TranslationPipeline, FeatureExtractionPipeline, MT5EncoderModel, AutoConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import TFAutoModel

from model_mT5_contrastive import mT5Classifier


def embed_sentence(in_path: str, out_path: str, finetuned: bool=False, batch_size:int=32):

    if finetuned:
        model = mT5Classifier(load_model_path="../experiments/dump/mt5_sup_chi")
        model = model.model
    else:
        model_name = "google/mt5-small"
        base_model = models.Transformer(model_name)
        # sentence bert repo does not current support mT5 yet (only T5 ironically)
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        mT5_config = AutoConfig.from_pretrained(model_name)
        base_model.auto_model = MT5EncoderModel.from_pretrained("google/mt5-small", config=mT5_config)
        pooling_model = models.Pooling(base_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[base_model, pooling_model])

    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kwargs = {'truncation': True, 'max_length': 512, 'return_text': True, "clean_up_tokenization_spaces": True}
    if in_path.endswith(".txt"):
        dataset = datasets.load_dataset('text', data_files={'train': [in_path]})["train"]
    elif in_path.endswith(".csv"):
        dataset = datasets.load_dataset('csv', data_files={'train': [in_path]})["train"]
        dataset = dataset.rename_column("Text", "text")
    elif ".jsonl" in in_path:
        dataset = datasets.load_dataset('text', data_files={'train': [in_path]})["train"]
        dataset = dataset.map(lambda example: {"content": json.loads(example["text"])["text"]}, remove_columns=["text"])
        dataset = dataset.rename_column("content", "text")
    else:
        raise NotImplementedError()

    embeddings = model.encode(dataset["text"], batch_size=batch_size, show_progress_bar=True, device="cpu")
    np.save(out_path, embeddings)


if __name__ == "__main__":
    data_dir='/projectnb2/cs505/students/lseoane/malawi_news_classification/data/test_chich_splits_text.csv'
    out_dir='/projectnb2/cs505/students/lseoane/malawi_news_classification/data/FINAL_test_chich'
    embed_sentence(data_dir,out_dir, finetuned=False, batch_size=16)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-in_file',
    #                     type=str,
    #                     help='the path to translation source file (in English)')
    # parser.add_argument('-out_file',
    #                     type=str,
    #                     help='the path to translation target file (in Nyanja) aka output')
    # parser.add_argument('-batch_size',
    #                     type=int,
    #                     help='batch_size for pipeline')
    #
    # args = parser.parse_args()
    #
    # embed_sentence(args.in_file, args.out_file, args.batch_size)
