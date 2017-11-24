# @Author: dileep
# @Last Modified by:   dileep

from collections import OrderedDict
import os
from typing import Tuple, Iterable, Sequence, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from . import datapath
from ..preprocessing import Encoder
from ..sampling import hold_out

class Churn:
    """
        Class for loading the `churn` dataset to predict whether customer `exited` or not
        Parameters:
        ----------
        features : Iterable[str]
            List of features to be used in training and testing.
            NOTE: Do not include the dependent variable
            Options: {RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,
                      Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,
                      EstimatedSalary}
        Attributes:
        ----------
        raw_data : pd.Series
            Raw data returned in the form of a pandas dataframe
        train_data : Tuple[np.ndarray, np.ndarray]
            Tuple of (features, targets) where each is a numpy ndarray
        test_data : Tuple[np.ndarray, np.ndarray]
            Tuple of (features, targets) where each is a numpy ndarray
    """
    _feature_dict = {
        'multi-category': {'Geography'},
        'binary-category': {'Gender', 'HasCrCard', 'IsActiveMember', 'Exited'},
        'int': {'CreditScore', 'Age', 'Tenure', 'NumofProducts'},
        'float': {'Balance', 'EstimatedSalary'}
    }

    def __init__(self, features: Iterable[str]) -> None:
        churn_path = os.path.join(datapath(), 'churn/Churn_Modeling.csv')
        assert self._validate_features(features), "Invalid features given"
        self.raw_data = pd.read_csv(churn_path)
        self._features = features + ['Exited']

    def __call__(self):
        raw_train, raw_test = hold_out(self.raw_data[self._features])
        feat_meta = self._get_feat_meta(self._features)
        data_encoder = Encoder(feat_meta)
        return data_encoder.encode(raw_train, raw_test, 'Exited')

    def _validate_features(self, features: Iterable[str]) -> bool:
        """
            Returns whether the input set of features are valid
            Parameters:
            ----------
            features : Iterable[str]
                Features input to the class
            Returns:
            -------
            bool
                True/False based on validity
        """
        all_features = set()
        for f_set in self._feature_dict.values():
            all_features.update(f_set)
        return not any(filter(lambda f: f not in all_features, features))

    def _get_feat_meta(self, features: Iterable[str]) -> Dict[str, str]:
        """
            Returns the type for each feature
            Parameters:
            ----------
            features : Iterable[str]
                A list of features that are to be used for classification
            Returns:
            -------
            Dict[str, str]
                Dictionary of features and their corresponding types
        """
        invert_fdict = {frozenset(v): k for k, v in self._feature_dict.items()}
        feat_meta: Dict[str, str] = OrderedDict()
        for feat in features:
            for feat_group, data_type in invert_fdict.items():
                if feat in feat_group:
                    feat_meta[feat] = data_type
                    continue
        return feat_meta

    def encode_features(self, features: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        cat_features = (self._feature_dict['binary-category'] or
                        self._feature_dict['multi-category'])
        for feat in features:
            if feat in cat_features:
                self.pp

    def split_data(self, features: Iterable[str]) -> Sequence[np.ndarray]:
        """
            Splits the raw data into training and testing using the features as a filter
            Parameters:
            ----------
            features : Iterable[str]
                Features that are to be used in the training and testing data
            Returns:
            -------
            Sequence[np.ndarray]
                Sequence of x_train, x_test, y_train, y_test
        """
        pass
