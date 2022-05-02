import json
import os
import argparse

from tqdm import tqdm
import datasets

from transformers import TranslationPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines.pt_utils import KeyDataset


def translate(in_path: str, out_path: str, batch_size:int=32, en_to_ny:bool = True):
    model_name = "Helsinki-NLP/opus-mt-en-ny" if en_to_ny else "Helsinki-NLP/opus-mt-ny-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = TranslationPipeline(model, tokenizer=tokenizer, device=0, batch_size=batch_size)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kwargs = {'truncation': True, 'max_length': 512, 'return_text': True, "clean_up_tokenization_spaces": True}
    if in_path.endswith(".csv"):
        dataset = datasets.load_dataset('csv', data_files={'train': [in_path]})["train"]
        dataset = dataset.rename_column("Text", "text")
    elif in_path.endswith(".txt"):
        dataset = datasets.load_dataset('text', data_files={'train': [in_path]})["train"]
    elif ".jsonl" in in_path:
        dataset = datasets.load_dataset('text', data_files={'train': [in_path]})["train"]
        dataset = dataset.map(lambda example: {"content": json.loads(example["text"])["text"]}, remove_columns=["text"])
        dataset = dataset.rename_column("content", "text")
    with open(out_path, "w") as out_f:
        pbar = tqdm(total=len(dataset))
        for out in pipe(KeyDataset(dataset, "text"), **kwargs):
            out_f.writelines(out[0]["translation_text"] + "\n")
            pbar.update(1)


if __name__ == "__main__":
    # translate("../data/english_news/coca-samples-text/text_mag.txt", "./test.txt")
    # translate("../data/english_news/realnews/realnews.jsonl9050", "./test.txt")
    translate("../data/train.csv", "../data/train_helsinki_translated.txt", batch_size=16, en_to_ny=False)
    # parser = argparse.ArgumentParser(description='')
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
    # translate(args.in_file, args.out_file, args.batch_size)
