import numpy as np
import pandas as pd 
from train_baseline_chichewa import * 


def results(df) -> None: 
    
    raise NotImplementedError

if __name__ == "__main__":
    chichewa_result = train_and_evaluate(data_path="../data/train.csv")
    chichewa_result.to_csv('../results/chichewa_eval_final.csv')


    