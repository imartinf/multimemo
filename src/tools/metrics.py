import math
import numpy
import pandas
import torch
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from typing import Union

Array = numpy.array
Series = pandas.Series
Tensor = torch.Tensor

def calc_pearson(y: Union[Tensor, Array, Series],
                 y_hat: Union[Tensor, Array, Series]) -> float:
    """ Pearson's Correlation coefficient between two sets.

    Args:
        y: (N,) Actual reference values.
        y_hat: (N,) Predicted values

    Returns:
        Index's value between [-1, 1].
    """
    score, p_value = pearsonr(y, y_hat)
    if math.isnan(score):
        return 0
    else:
        return score


def calc_spearman(y: Union[Tensor, Array, Series],
                 y_hat: Union[Tensor, Array, Series]) -> float:
    """ Spearman's Rank Correlation coefficient between two sets.

    Args:
        y: (N,) Actual reference values.
        y_hat: (N,) Predicted values

    Returns:
        Index's value between [-1, 1].
    """
    score, p_value = spearmanr(y, y_hat)
    if math.isnan(score):
        return 0
    else:
        return score