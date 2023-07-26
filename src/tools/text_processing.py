import numpy as np

def extract_sentence_embeddings(data, column, out, tokenizer, model):
    '''
    Extracts sentence embeddings from a given model. should return a dataframe with a new column storing the embeddings

    data: pandas dataframe with a text column defined by the argument column
    column: string with the name of the column containing the text
    out: string with the name of the column to store the embeddings
    tokenizer: tokenizer to use
    model: model to use. It should be a sentence transformer model
    '''

    data = data.copy()
    data[out] = data[column].apply(lambda x: model.encode(x, tokenizer))
    return data

def compute_cosine_similarity(data, column1, column2, out):
    '''
    Computes the cosine similarity between two vectors. Should return a dataframe with a new column storing the cosine similarity
    
    data: pandas dataframe with two columns containing the vectors
    column1: string with the name of the first column containing the vectors
    column2: string with the name of the second column containing the vectors
    out: string with the name of the column to store the cosine similarity
    '''

    data = data.copy()
    data[out] = data.apply(lambda x: np.dot(x[column1], x[column2])/(np.linalg.norm(x[column1])*np.linalg.norm(x[column2])), axis=1)
    return data

def filter_embeddings(data, column1, column2, sim, out, strategy, threshold=0):
    '''
    Given a dataset with two text columns and corresponding sentence embeddings, copies the text in the text in the first column
    to the text in the second column if the similarity between the embeddings is below a threshold. Should return a dataframe
    with a new column storing the filtered text.

    data: pandas dataframe with two columns containing the text and two columns containing the embeddings
    column1: string with the name of the first column containing the text
    column2: string with the name of the second column containing the text
    sim: string with the name of the column containing the similarity between the embeddings
    out: string with the name of the column to store the filtered text
    strategy: string with the strategy to use. It can be 'threshold' or 'iqr'
    threshold: float with the threshold to use. Only used if strategy is 'threshold' 
    '''

    data = data.copy()
    if strategy == 'threshold':
        data[out] = data.apply(lambda x: x[column1] if x[sim] < threshold else x[column2], axis=1)
    elif strategy == 'iqr':
        q1 = data[sim].quantile(0.25)
        q3 = data[sim].quantile(0.75)
        iqr = q3 - q1
        data[out] = data.apply(lambda x: x[column1] if x[sim] < q1 - 1.5*iqr else x[column2], axis=1)
    else:
        raise ValueError("Invalid strategy")
    return data

def apply_similarity_filter(data, tokenizer, model, column1, column2, sim, out, strategy, threshold=0):
    '''
    Applies the similarity filter pipeline to a dataset. Should return a dataframe with a new column storing the filtered text
    '''

    data = data.copy()
    data = extract_sentence_embeddings(data, column1, 'embeddings1', tokenizer, model)
    data = compute_cosine_similarity(data, column1, column2, sim)
    data = filter_embeddings(data, column1, column2, sim, out, strategy, threshold)
    return data