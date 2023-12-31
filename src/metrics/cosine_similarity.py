from src.metrics.base_text_metric import BaseTextMetric
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity(BaseTextMetric):
    """
    Cosine Similarity metric
    """

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def get_metric(self, texts1, texts2, labels=None, *args, **kwargs):
        """
        Calculate the cosine similarity between two texts.
        """
        if self.model == "precomputed" and "paths" in kwargs:
            embeddings1 = self.get_embeddings(texts1, path=kwargs['paths'][0], *args, **kwargs)
            embeddings2 = self.get_embeddings(texts2, path=kwargs['paths'][1], *args, **kwargs)
        else:
            embeddings1 = self.get_embeddings(texts1, *args, **kwargs)
            embeddings2 = self.get_embeddings(texts2, *args, **kwargs)
        # Check shape
        if embeddings1.shape != embeddings2.shape:
            raise ValueError(f"Embeddings must have the same shape. Got {embeddings1.shape} and {embeddings2.shape}.")
        embeddings1 = embeddings1.reshape(1, -1)
        embeddings2 = embeddings2.reshape(1, -1)
        return cosine_similarity(embeddings1, embeddings2)

    