import re

import pandas as pd
from tqdm import tqdm
from typing import List, Optional


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [s for s in sentences if s]


def expand_sentences(
    texts: List[str],
    translated_texts: List[str],
    labels: Optional[List[str]]=None,
    sents_per_split: int=3,
    method: str="interpolate"
):
    new_texts = []
    new_translated_texts = []
    new_labels = []
    news_idx = []
    for i in tqdm(range(len(texts))):
        text_split = split_sentences(texts[i])
        translated_text_split = split_sentences(translated_texts[i])
        if method == "interpolate":
            prev_translated_idx = 0
            len_matched = len(text_split) == len(translated_text_split)
            cur_news_idx = []
            for sent_idx in range(0, len(text_split), sents_per_split):
                translated_idx = sent_idx + sents_per_split if len_matched else \
                    round((sent_idx+3) / len(text_split) * len(translated_text_split))
                new_texts.append(" ".join(text_split[sent_idx: sent_idx+sents_per_split]))
                new_translated_texts.append(" ".join(translated_text_split[prev_translated_idx: translated_idx]))
                if labels is not None:
                    new_labels.append(labels[i])
                prev_translated_idx = translated_idx
                cur_news_idx.append(i)
        elif method == "s2s":
            cur_news_idx = [i]
            if len(text_split) > len(translated_text_split):
                new_texts.append(" ".join(text_split[:len(translated_text_split)]))
                new_translated_texts.append(" ".join(translated_text_split))
            else:
                new_translated_texts.append(" ".join(translated_text_split[:len(text_split)]))
                new_texts.append(" ".join(text_split))
        else:
            raise NotImplementedError()
        news_idx.extend(cur_news_idx)
    if labels is None:
        return new_texts, new_translated_texts
    else:
        return new_texts, new_translated_texts, new_labels, news_idx


