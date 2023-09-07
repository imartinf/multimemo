from matplotlib import pyplot as plt
import pandas as pd
from tqdm import trange


def plot_histogram(data, columns, title, xlabel, ylabel, bins=10, figsize=(9, 5), show=True, save_path=None):
    plt.figure(figsize=figsize)
    for column in columns:
        plt.hist(data[column], bins=bins, alpha=0.5, label=column, density=False)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_lineplot(data, x, y, title, xlabel, ylabel, figsize=(9, 5), err=None, show=True, save_path=None):
    plt.figure(figsize=figsize)
    plt.plot(data[x], data[y])
    if err is not None:
        plt.errorbar(data[x], data[y], yerr=data[err], fmt='o', capsize=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_oov_vs_cos_sim(data,col,show=True,save_path=None):
    cosine_sim_vs_oov_df = pd.DataFrame(columns=['oov', 'mean', 'std'])
    for oov in trange(0, data[col].max() + 1):
        cosine_sim_vs_oov_df = pd.concat([cosine_sim_vs_oov_df, pd.DataFrame({
            'oov': oov,
            'mean': data[data['recaption_oov_words'] == oov]['cosine_sim_mpnet'].mean(),
            'std': data[data['recaption_oov_words'] == oov]['cosine_sim_mpnet'].std()
        }, index=[0])], ignore_index=True)

    plt.figure(figsize=(9, 5))
    plt.errorbar(cosine_sim_vs_oov_df['oov'], cosine_sim_vs_oov_df['mean'], yerr=cosine_sim_vs_oov_df['std'], fmt='o', capsize=5)
    plt.title("Cosine similarity vs OOV words")
    plt.xlabel("Number of OOV words")
    plt.ylabel("Cosine similarity")
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    