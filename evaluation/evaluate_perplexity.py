from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_perplexity(model, tokenizer, model_input: List[str], stride: int = 512) -> torch.Tensor:
    # source: https://huggingface.co/docs/transformers/perplexity
    max_length = model.config.n_positions
    stride = min(max_length, stride)

    encodings = tokenizer(model_input, return_tensors="pt", padding="longest", truncation=True)

    nlls = []
    # if sentences are longer than default window size
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        # padded tokens set to -100, no attention no loss https://github.com/huggingface/transformers/issues/2630
        target_ids[target_ids == tokenizer.vocab[tokenizer.pad_token]] = -100
        # switch it to EOS because model word embedding doesn't have EOS. As long as label is -100 what token it
        # switches to doesn't impact performance
        input_ids[input_ids == tokenizer.vocab[tokenizer.pad_token]] = tokenizer.eos_token_id

        with torch.no_grad():
            # instead of taking aggregated cross entropy from causal LM, we calculate
            # per sentence without reduction.
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L1072
            outputs = model(input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.view(shift_labels.shape).sum(dim=1)

        nlls.append(neg_log_likelihood)

    sent_lens = encodings.attention_mask.sum(dim=1).to(model.device)
    ppl = torch.exp(torch.stack(nlls).sum(dim=0) / sent_lens)
    return ppl


def evaluate_perplexity(sentences: List[str], batch_size: int=2) -> np.array:
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device="cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    perplexities = np.array([])

    with torch.no_grad():
        for e in tqdm((range(0, len(sentences), batch_size))):
            ppl = get_perplexity(model, tokenizer, sentences[e: e + batch_size])
            perplexities = np.append(perplexities, ppl.cpu().numpy())
    return perplexities


def add_perplexity_to_dataset(path: str):
    df = pd.read_csv(path)
    if "perplexity" not in df.columns:
        df["Text"] = df["Text"].fillna(" ")
        df["perplexity"] = evaluate_perplexity(list(df["Text"]))
        df.to_csv(path, index=False)


if __name__ == "__main__":
    add_perplexity_to_dataset("../data/test_webtran_translated.csv")
    add_perplexity_to_dataset("../data/train_webtran_translated.csv")
    add_perplexity_to_dataset("../data/test_google_translated.csv")
    add_perplexity_to_dataset("../data/train_google_translated.csv")


