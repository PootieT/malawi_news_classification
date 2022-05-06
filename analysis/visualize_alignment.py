

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


def get_cmap(n, name='prism'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def visualize_alignment(emb_ny: np.array, emb_en: np.array, train_df: pd.DataFrame, title: str):
    emb_all = np.vstack([emb_ny, emb_en])
    pca = PCA(n_components=2)
    result = pca.fit_transform(emb_all)
    # create a scatter plot of the projection
    plt.figure(figsize=(15, 12))

    # plt.scatter(result[:, 0], result[:, 1])

    color_labels = train_df.Label
    unique_color_labels = list(set(color_labels))
    cmap = get_cmap(len(unique_color_labels))
    color2idx = {c: i for i, c in enumerate(unique_color_labels)}

    for i, clabel in enumerate(color_labels):
        # plt.annotate(label, xy=(result[i, 0], result[i, 1]))
        plt.plot(result[i, 0], result[i, 1], "o", c=cmap(color2idx[clabel]))
        plt.plot(result[i+len(train_df), 0], result[i+len(train_df), 1], "x", c=cmap(color2idx[clabel]))

    # plt.show()
    legend_patches = [mpatches.Patch(color=cmap(color2idx[c]), label=c) for c in unique_color_labels]

    plt.legend(handles=legend_patches)
    plt.title(title)
    plt.savefig(f"{title}.png")


if __name__ == "__main__":
    emb_chichewa = np.load("../data/mt5_embeddings/embeddings_chichewa_512_final.npy")
    emb_english = np.load("../data/mt5_embeddings/embeddings_english_512_final.npy")
    # emb_chichewa = np.load("../data/aligned_mt5_embeddings/embeddings_chichewa.npy")
    # emb_english = np.load("../data/aligned_mt5_embeddings/embeddings_english.npy")
    # emb_chichewa = np.load("../data/aligned_mt5_embeddings_no_dense/embeddings_chichewa.npy")
    # emb_english = np.load("../data/aligned_mt5_embeddings_no_dense/embeddings_english.npy")
    train_df = pd.read_csv("../data/train.csv")
    visualize_alignment(emb_chichewa, emb_english, train_df, title="mt5 embedding visualization")