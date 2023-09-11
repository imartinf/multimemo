from collections import Counter
import os
import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tiktoken import Encoding

from src.visualization.visualize import plot_histogram

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def remove_punctuation(text):
    """
    Remove punctuation from a text.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def extract_oov_words(data, oov_metric, in_columns, out_columns, save_paths=None):
    for (in_col, out_col) in zip(in_columns, out_columns):
        data[out_col] = oov_metric.get_metric(data[in_col].values, notebook=False)
        data[out_col + '_ratio'] = data[out_col] / data[in_col].str.split().str.len()
        print(f"Successfully computed OOV for {len(data)} {in_col}.")
        print(f"Sample {in_col} OOV: {data[out_col].values[:5]}")
        print(f"Sample {in_col} OOV ratio: {data[out_col + '_ratio'].values[:5]}")
        print(f"Average {in_col} OOV ratio: {np.mean(data[out_col + '_ratio'].values)}")

def extract_unk_tokens(data, unk_metric, in_columns, out_columns, save_paths=None):
    for (in_col, out_col) in zip(in_columns, out_columns):
        data[out_col] = unk_metric.get_metric(data[in_col].values, notebook=False)
        data[out_col + '_ratio'] = data[out_col] / data[in_col].str.split().str.len()
        print(f"Successfully computed UNK for {len(data)} {in_col}.")
        print(f"Sample {in_col} UNK: {data[out_col].values[:5]}")
        print(f"Sample {in_col} UNK ratio: {data[out_col + '_ratio'].values[:5]}")
        print(f"Average {in_col} UNK ratio: {np.mean(data[out_col + '_ratio'].values)}")

    base, ext = os.path.splitext(save_paths)
    save_paths_ratio = (base + '_ratio' + ext)
    plot_histogram(data, out_columns, title="OOV words", xlabel="Number of OOV words", ylabel="Number of texts", bins=10, figsize=(9, 5), show=False, save_path=save_paths)
    plot_histogram(data, [out + '_ratio' for out in out_columns], title="OOV words ratio", xlabel="OOV words ratio", ylabel="Number of texts", bins=20, figsize=(9, 5), show=False, save_path=save_paths_ratio)
    return data

def most_frequent_oov_words(data, col, oov_metric, nb_words=20):
    tokenizer = oov_metric.tokenizer
    oov_words = Counter()
    for _,row in tqdm(data.iterrows(), total=len(data)):
        for word in row[col].split():
            if oov_metric.is_oov(word):
                oov_words[word] += 1

    print(f"Number of OOV words in {col}: ", len(oov_words))

    oov_words = oov_words.most_common(nb_words)
    oov_tokens = {}
    for oov_word, _ in oov_words:
        if isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast):
            tokens = tokenizer.tokenize(oov_word)
        elif isinstance(tokenizer, Encoding):
            tokens = [tokenizer.decode_single_token_bytes(token) for token in tokenizer.encode(oov_word)]
        if len(tokens) > 1:
            tokens = [token for token in tokens if token not in ['[UNK]', '[PAD]']]
        if len(tokens) > 0:
            oov_tokens[oov_word] = tokens
    print("20 Most frequent OOV words split into tokens: ")
    # Print most frequent
    print(oov_tokens)
    return oov_tokens

def most_frequent_unk_tokens(data, col, unk_metric, nb_words=20):
    tokenizer = unk_metric.tokenizer
    unk_tokens = Counter()
    for _,row in tqdm(data.iterrows(), total=len(data)):
        for word in row[col].split():
            if tokenizer.encode(word, add_special_tokens=False).count(tokenizer.unk_token_id) > 0:
                unk_tokens[word] += 1

    print(f"Number of UNK tokens in {col}: ", len(unk_tokens))

    unk_tokens = unk_tokens.most_common(nb_words)
    print("20 Most frequent UNK tokens: ")
    # Print most frequent
    print(unk_tokens)
    return unk_tokens

def print_metric_examples(data, sort_col, data_cols, metric, nb_examples=10):
    for _,row in data.sort_values(by=sort_col, ascending=False).head(nb_examples).iterrows():
        print(f"{data_cols[0]}: {row[data_cols[0]]}")
        print(f"{data_cols[1]}: {row[data_cols[1]]}")
        print(f"Cosine similarity: {metric.get_metric([row[data_cols[0]]], [row[data_cols[1]]])[0][0]}")
        print()

def compute_cossim_wrt_oov(data, oov_col='recaption_oov_words', cos_sim_col='cosine_sim_mpnet'):
    result_df = pd.DataFrame(columns=['oov', 'mean', 'std'])
    for oov in trange(0, data[oov_col].max() + 1):
        result_df = pd.concat([result_df, pd.DataFrame({
            'oov': oov,
            'mean': data[data[oov_col] == oov][cos_sim_col].mean(),
            'std': data[data[oov_col] == oov][cos_sim_col].std()
        }, index=[0])], ignore_index=True)
    return result_df
