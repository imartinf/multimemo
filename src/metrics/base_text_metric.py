from sentence_transformers import SentenceTransformer

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
            return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True ,device=self.device)
        else:
            raise NotImplementedError
    
    def get_metric(self, texts1, texts2=None, labels=None):
        """
        Calculate the metric between two texts.
        """
        raise NotImplementedError
