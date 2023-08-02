import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from openai.embeddings_utils import get_embedding

class BaseTextMetric:
    """
    Base class for text metrics that involve extracting embeddings using a model and then calculating a metric.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_embeddings(self, texts):
        """
        Extract embeddings from a text using the model and tokenizer.
        """
        if isinstance(self.model, SentenceTransformer):
            return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True ,device=self.device)
        elif isinstance(self.model, str):
            if isinstance(texts, list):
                print(f"Warning: {self.model} only supports one text at a time.")
                print("Using the first text in the list.")
                texts = texts[0]
            response = get_embedding(texts, self.model)
            return np.array(response).reshape(1, -1)
        else:
            raise NotImplementedError
    
    def get_metric(self, texts1, texts2=None, labels=None):
        """
        Calculate the metric between two texts.
        """
        raise NotImplementedError
