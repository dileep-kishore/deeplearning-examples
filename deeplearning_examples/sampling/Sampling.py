# @Author: dileep
# @Last Modified by:   dileep

"""
    Module that contains various functions to split, sample and permute data
"""

from typing import Tuple, Optional, Iterable
import pandas as pd
from sklearn.model_selection import train_test_split

def hold_out(data_set: pd.DataFrame, frac: float = 0.20, shuffle: bool = True,
             stratify: Optional[Iterable] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Splits data according to frac into training and testing
        Parameters:
        ----------
        data_set : pd.DataFrame
            DataSet to be split
        frac : float, optional
            The fraction of data that will be used for testing
        shuffle : bool, optional
            Randomly shuffle the data before splitting
        Returns:
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of (train_data, test_data)
    """
    return train_test_split(data_set, test_size=frac, shuffle=shuffle, stratify=stratify)
