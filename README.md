# malawi_news_classification
Low resource text classification

Welcome to the repo for final class project for CS 505 (NLP). In this project
we are tasked with this [Malawi News Classification dataset](https://zindi.africa/competitions/ai4d-malawi-news-classification-challenge/leaderboard).
In limited time span, we tested a few techniques in data augmentation, creating / finetuning better embedding space
with Transformer-based models, as well as some data science techniques to boost performance in feature space.

## Baseline models:
For any baseline models.
- Support Vector Machines
- Random Forests
- XGBoost 
- Multi-layer Perceptron
- Logistic Regression

For Classification Results from all the models:
```bash
    python3.9 experiments/main.py -<data_dir> -<embedding_file>
```

- data_dir : Directory where the training data is located (Text)
- embedding_file : Name of the embedding file 



- The results will be generated as a csv file in this [location](Results)
    

## Data Augmentation methods:
- **Mixup**      - [Script](https://github.com/PootieT/malawi_news_classification/blob/main/models/mixUp.py)     
     ```bash
        python mixUp.py -<train_data_dir> -<embeddings type>
     ```

    - "Embeddings type" means the kind of embeddings to use when augmenting the data
    - Mixup Augmented data will be generated in this [Location]()

- **NLPAug**    - [Script](https://github.com/PootieT/malawi_news_classification/blob/main/experiments/Augmention_Proof-of-Concept%20.ipynb)     

    - [NLPAug Description](https://nlpaug.readthedocs.io/en/latest/) 
 
- **Manual News Scraping** - [Data](https://github.com/PootieT/malawi_news_classification/tree/main/data_gathering) 
    
## Types of embedding methods used: 
- Count Vectorization
- TFIDF
- English aligned Chichewa MT5 embeddings     - [Script](https://github.com/PootieT/malawi_news_classification/blob/main/experiments/train_mt5_contrastive.py)
    ```bash
    python train_mt5_contrastive.py
    ```

## Parallel RealNews Subset
For our alignment experiment, we created our own parallel news dataset.
To recreate such data, you need to:

1. Download realnews dataset from GROVER [Repo](https://github.com/rowanz/grover/tree/master/realnews)
2. Split files into smaller chunks for parallel translation (if running models) or small enough for Google Translation
   ```bash
   ./split_file_process_template.sh <input_path> <num_partition>
   ```
3. Translating the files!
   1. If you are running in SCC and translating with Marian English-Chichewa Translation Model, you can run
    ```bash
    qsub utils/run_translation_en_ny.qsub
    ```
   2. If you choose to use Google, the easiest free way is to convert them into chunks of excel sheets no bigger than 2 
      mb, and submit them as files manually. `utils` should have some file conversion file you may find helpful.
4. Once you obtain translation files (Or, check SCC `/projectnb/cs505/projects/realnews`), you can run alignment 
  training with:
  ```bash
  python experiments/train_mt5_contrastive.py
  ```
  (make sure you modify the paths to the Chichewa and English files in main section.)
