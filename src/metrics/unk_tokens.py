from tiktoken import Encoding
from src.metrics.base_text_metric import BaseTextMetric
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class UNKTokens(BaseTextMetric):
    """
    Compute the number of Unknown tokens for a given text set and a tokenizer.
    """

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def get_metric(self, texts1, texts2=None, labels=None, **kwargs):
        """
        Calculate the number of unknown tokens for a given text set and a tokenizer.
        """
        if labels is not None:
            raise NotImplementedError
        if texts2 is not None:
            raise NotImplementedError
        unk = []
        # Select between tqdm notebook and tqdm terminal depending on the notebook argument
        if kwargs.get("notebook", False):
            iterator = tqdm_notebook(texts1, desc="UNK Tokens")
        else:
            iterator = tqdm(texts1, desc="UNK Tokens")
        for text in iterator:
            # Compute UNK Tokens by adding up the number of UNK tokens in each text
            if isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast):
                unk.append(self.tokenizer.encode(text, add_special_tokens=False).count(self.tokenizer.unk_token_id))
            else:
                raise NotImplementedError
        return unk
