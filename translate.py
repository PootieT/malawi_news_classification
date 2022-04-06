import pandas as pd
import requests
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


def webtran_translate(text: List[str]) -> List[str]:
    url = "https://www.webtran.eu/gtranslate/"
    translations = []
    for sentence in tqdm(text):
        params = {
            "text": sentence,
            "gfrom": "ny",
            "gto": "en",
            "key": "ABC"
        }
        # headers = {"content-type"}
        try:
            translated_sent = requests.post(url, data=params).text
        except Exception as e:
            print(f"Exception translating line '{sentence}': {e}")
            translated_sent = sentence
        translations.append(translated_sent)
    return translations


def lingvanext_translate(text: List[str]) -> List[str]:
    url = "https://api-b2b.backenster.com/b1/api/v3/translate/"
    separator = "<break>"
    translations = []
    params = {
        "from": "ny_MW",
        "to": "en_GB",
        "platform": "dp"
    }
    cum_len = 0
    cum_text = ""
    for sentence in tqdm(text):
        if cum_len < 10000 and len(sentence) + cum_len < 10000:
            cum_len += len(sentence) + len(separator)
            cum_text += f"{separator} {sentence}"
        else:
            params["text"] = cum_text
            try:
                #TODO need bearer token or something
                translated_text = requests.post(url, data=params).json().get("result")
            except Exception as e:
                print(f"Exception translating {e}")
                translated_text = cum_text
            translations.extend(translated_text.split(separator))
    return translations


if __name__ == "__main__":
    test_df = pd.read_csv("data/test.csv")
    # test_df["Text"] = webtran_translate(test_df["Text"])
    test_df["Text"] = lingvanext_translate(test_df["Text"])
    test_df.to_csv("data/test_webtran_translated.csv")

    train_df = pd.read_csv("data/train.csv")
    # train_df["Text"] = webtran_translate(train_df["Text"])
    train_df["Text"] = lingvanext_translate(train_df["Text"])
    train_df.to_csv("data/train_webtran_translated.csv")