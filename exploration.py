import pandas as pd

def main():
    df = pd.read_csv("data/Train.csv")
    df.describe()
    df["Label"].value_counts()
    df["Text"].apply(len).describe()
    pass


if __name__ == "__main__":
    main()