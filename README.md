# malawi_news_classification
Low resource text classification

Welcome to the repo for final class project for CS 505 (NLP). In this project
we are tasked with this [Malawi News Classification dataset](https://zindi.africa/competitions/ai4d-malawi-news-classification-challenge/leaderboard).
In limited time span, we tested a few techniques in data augmentation, creating / finetuning better embedding space
with Transformer-based models, as well as some data science techniques to boost performance in feature space.

## Baseline models:
For any baseline models.

## Data Augmentation methods:
- **Mixup**      - [Script](https://github.com/PootieT/malawi_news_classification/blob/main/models/mixUp.py)     
     ```bash
        python3.9 mixUp.py -<train_data_dir> -<embeddings type>
     ```

    - "Embeddings type" means the kind of embeddings to use when augmenting the data
    - Mixup Augmented data will be generated in this [Location]()

- **NLPAug**    - [Script](https://github.com/PootieT/malawi_news_classification/blob/main/experiments/Augmention_Proof-of-Concept%20.ipynb)     

    - [NLPAug Description](https://nlpaug.readthedocs.io/en/latest/) 
 
- **Manual News Scraping** - [Data](https://github.com/PootieT/malawi_news_classification/tree/main/data_gathering) 
    
## For Embedding Generation

### Types of embedding methods used: 
- Count Vectorization
- TFIDF
- English aligned Chichewa MT5 embeddings     - [Script](https://github.com/PootieT/malawi_news_classification/blob/main/experiments/train_mt5_contrastive.py)
    ```bash
    python3.9 train_mt5_contrastive.py
    ```


