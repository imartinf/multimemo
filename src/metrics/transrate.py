import numpy as np
from src.metrics.base_text_metric import BaseTextMetric

class TransRate(BaseTextMetric):
    """
    Implementation of the TransRate metric from:

    Huang, L. K., Huang, J., Rong, Y., Yang, Q., & Wei, Y. (2022, June).
    Frustratingly easy transferability estimation.
    In International Conference on Machine Learning (pp. 9201-9225). PMLR.
    """

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def _rate_distortion(self, Z, nc=1, eps=1e-4):
        """
        Calculate the rate-distortion for a given set of embeddings and epsilon.
        """
        (d,n) = Z.shape
        (_, rate) = np.linalg.slogdet(np.eye(d) + (1/(n*eps)*Z.T@Z))
        return 0.5*rate
        

    def get_metric(self, texts1, texts2=None, labels=None, **kwargs):
        """
        Calculate the TransRate metric of a text set.

        TODO: Project variance matrix by a matrix of centroids of each class.
        """
        # Get embeddings
        Z = self.get_embeddings(texts1)
        # Centralize embeddings
        Z = Z - np.mean(Z, axis=0, keepdims=True)
        RZ = self._rate_distortion(Z)

        RZY = 0

        # Rank labels and divide in 10 ranges
        if labels is None:
            labels = np.arange(len(texts1))
        labels = np.array(labels)
        # Sort labels and embeddings by label rank
        sorted_idx = np.argsort(labels)
        y = labels[sorted_idx]
        Z = Z[sorted_idx]

        K = int(y.max() + 1)
        for i in range(K):
            RZY += self._rate_distortion(Z[y==i.flatten()])
        return RZ - RZY/K