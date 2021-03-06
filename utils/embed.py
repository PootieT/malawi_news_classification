import json
import os

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, models
import datasets

import torch
from transformers import MT5EncoderModel, AutoConfig
from models.model_mT5_contrastive import mT5Classifier


def embed_sentence(
    in_path: str,
    out_path: str,
    finetuned: bool = False,
    load_dense_layer: bool = True,
    batch_size: int = 32,
):
    if finetuned:
        model = mT5Classifier(
            load_model_path="../experiments/dump/mt5_sup_chi",
            load_dense_layer=load_dense_layer,
        )
        model = model.model
    else:
        model_name = "google/mt5-small"
        base_model = models.Transformer(model_name)
        # sentence bert repo does not current support mT5 yet (only T5 ironically)
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        mT5_config = AutoConfig.from_pretrained(model_name)
        base_model.auto_model = MT5EncoderModel.from_pretrained(
            "google/mt5-small", config=mT5_config
        )
        pooling_model = models.Pooling(base_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[base_model, pooling_model])

    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if in_path.endswith(".txt"):
        dataset = datasets.load_dataset("text", data_files={"train": [in_path]})[
            "train"
        ]
    elif in_path.endswith(".csv"):
        dataset = datasets.load_dataset("csv", data_files={"train": [in_path]})["train"]
        dataset = dataset.rename_column("Text", "text")
    elif ".jsonl" in in_path:
        dataset = datasets.load_dataset("text", data_files={"train": [in_path]})[
            "train"
        ]
        dataset = dataset.map(
            lambda example: {"content": json.loads(example["text"])["text"]},
            remove_columns=["text"],
        )
        dataset = dataset.rename_column("content", "text")
    else:
        raise NotImplementedError()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = model.encode(
        dataset["text"], batch_size=batch_size, show_progress_bar=True, device=device
    )
    np.save(out_path, embeddings)


if __name__ == "__main__":
    # embed_sentence("../data/train.csv", "../data/aligned_mt5_embeddings_no_dense/embeddings_chichewa.npy",
    #                finetuned=True, load_dense_layer=False, batch_size=16)
    # embed_sentence("../data/test.csv", "../data/aligned_mt5_embeddings_no_dense/test_embeddings_chichewa.npy",
    #                finetuned=True, load_dense_layer=False, batch_size=16)
    # embed_sentence("../data/test_google_translated.csv", "../data/mt5_embeddings/test_embeddings_english.npy",
    #                finetuned=False, load_dense_layer=True, batch_size=16)
    # embed_sentence("../data/train_google_translated.csv", "../data/aligned_mt5_embeddings_no_dense/embeddings_english.npy",
    #                finetuned=True, load_dense_layer=False, batch_size=16)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--in_file", type=str, help="the path to translation source file (in English)"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        help="the path to translation target file (in Nyanja) aka output",
    )
    parser.add_argument("--batch_size", type=int, help="batch_size for pipeline")
    parser.add_argument(
        "--finetuned", action="store_true", help="to use finetuned model or not"
    )
    parser.add_argument(
        "--load_dense_layer",
        action="store_true",
        help="to load last dense projection layer in fintuned model or not",
    )

    args = parser.parse_args()

    embed_sentence(
        args.in_file,
        args.out_file,
        batch_size=args.batch_size,
        finetuned=args.finetuned,
        load_dense_layer=args.load_dense_layer,
    )
