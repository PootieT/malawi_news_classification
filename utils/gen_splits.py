#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import csv
import json
import logging
import math
from typing import List, Dict, Tuple, Callable, Iterable, Optional, Any

import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from numpy import save
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization

from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator, SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import MT5EncoderModel, AutoConfig,T5Tokenizer
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import re


# In[22]:


# train_df=pd.read_csv('C:/Users/leose/homework/malawi_news_classification/data/train_google_translated.csv')
train_df=pd.read_csv('C:/Users/leose/homework/malawi_news_classification/data/train.csv')

# model = MT5EncoderModel.from_pretrained("google/mt5-small")
# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")


# In[23]:


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
                translated_idx = sent_idx + sents_per_split if len_matched else                     round((sent_idx+3) / len(text_split) * len(translated_text_split))
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


# In[24]:




def expand_mono_sentences(
    texts: List[str],
    labels: Optional[List[str]]=None,
    sents_per_split: int=3,
    method: str="interpolate"
):
    new_texts = []
    new_labels = []
    news_idx = []
    
    for i in tqdm(range(len(texts))):
        text_split = split_sentences(texts[i])
        if method == "interpolate":
            prev_translated_idx = 0
            len_matched = len(text_split)
            cur_news_idx = []
            for sent_idx in range(0, len(text_split), sents_per_split):
                translated_idx = sent_idx + sents_per_split if len_matched else                     round((sent_idx+3) / len(text_split) * len(translated_text_split))
                new_texts.append(" ".join(text_split[sent_idx: sent_idx+sents_per_split]))
                if labels is not None:
                    new_labels.append(labels[i])
                prev_translated_idx = translated_idx
                cur_news_idx.append(i)
        elif method == "s2s":
            cur_news_idx = [i]
            if len(text_split) > len(translated_text_split):
                new_texts.append(" ".join(text_split[:len(translated_text_split)]))
            else:
                new_texts.append(" ".join(text_split))
        else:
            raise NotImplementedError()
        news_idx.extend(cur_news_idx)
    if labels is None:
        return new_texts
    else:
        return new_texts, new_labels, news_idx


# In[25]:


# input_ids = []

# splits_text,splits_translated,splits_labels,split_index=expand_sentences(train_df1['Text'].tolist(),train_df2['Text'].tolist(),train_df['Label'].tolist())
splits_text,splits_labels,split_index=expand_mono_sentences(train_df['Text'].tolist(),train_df['Label'].tolist())
split_text_df=pd.DataFrame(splits_text,columns=['Text'])
split_text_df.to_csv('./data/' + 'aligned_chich_split_texts' + '.csv')

split_labels=pd.DataFrame(splits_labels,columns=['Label'])
split_labels.to_csv('./data/' + 'aligned_chich_split_labels' + '.csv')


split_index=pd.DataFrame(split_index,columns=['Idx'])
split_index.to_csv('./data/' + 'aligned_chich_split_index' + '.csv')

# split_trans_df=pd.DataFrame(splits_translated,columns=['Text'])
# split_trans_df.to_csv('C:/Users/leose/homework/malawi_news_classification/data/' + 'eng_and_chich_split_translated_text' + '.csv')


# In[ ]:




