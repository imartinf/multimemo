from src.metrics.base_text_metric import BaseTextMetric
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


class OOVWords(BaseTextMetric):
    """
    Compute the number of Out Of Vocabulary Words for a given text set and a tokenizer.
    """

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def is_oov(self, word):
        """
        Check if a word is Out Of Vocabulary.
        """
        return len(self.tokenizer.tokenize(word)) > 1

    def get_metric(self, texts1, texts2=None, labels=None, **kwargs):
        """
        Calculate the number of Out Of Vocabulary Words for a given text set and a tokenizer.
        Return a list that contains the number of OOV for each text.
        """
        if labels is not None:
            raise NotImplementedError
        if texts2 is not None:
            raise NotImplementedError
        oov = []
        # Select between tqdm notebook and tqdm terminal depending on the notebook argument
        if kwargs.get("notebook", False):
            iterator = tqdm(texts1, desc="OOV words")
        else:
            iterator = tqdm_notebook(texts1, desc="OOV words")
        for text in iterator:
            # Compute OOV words by checking if each word is present in the tokenizer vocabulary
            oov.append(sum([self.is_oov(word) for word in text.split()]))
        return oov
