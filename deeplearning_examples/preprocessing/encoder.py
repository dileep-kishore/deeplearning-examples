# @Author: dileep
# @Last Modified by:   dileep

"""
    Normalize, scale and encode features in your dataset
"""

from collections import OrderedDict
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

EncodedData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

class Encoder:
    """
        A class that normalizes, scales and encodes features in the dataset using metadata
        Parameters:
        ----------
        metadata : Dict[str, str]
            An ordered dictionary containing information about the type of data in each feature
    """
    #TODO: Verify the distribution of values in the features and then select rules
    _encoder_rules = {
        'float': StandardScaler,
        'int': StandardScaler,
        'binary-category': LabelEncoder,
        'multi-category': [LabelEncoder, OneHotEncoder]
    }

    def __init__(self, metadata: Dict[str, str]) -> None:
        self.preprocess_dict: Dict[str, Any] = OrderedDict()
        for feat, feat_type in metadata.items():
            self.preprocess_dict[feat] = self._encoder_rules[feat_type]

    def _encode_x(self, train: pd.DataFrame, test: pd.DataFrame):
        """ Encode independent variables """
        train_encoded = np.zeros(train.shape)
        test_encoded = np.zeros(test.shape)
        onehots = []
        for i, feat in enumerate(train.columns):
            processor = self.preprocess_dict[feat]
            if isinstance(processor, list):
                #FIXME: Assuming this involves only the OneHotEncoder for now
                processor_inst = processor[0]()
                train_encoded[:, i] = processor_inst.fit_transform(train[feat].values.reshape(-1, 1)).squeeze()
                test_encoded[:, i] = processor_inst.transform(test[feat].values.reshape(-1, 1)).squeeze()
                onehots.append(i)
            else:
                processor_inst = processor()
                train_encoded[:, i] = processor_inst.fit_transform(train[feat].values.reshape(-1, 1)).squeeze()
                test_encoded[:, i] = processor_inst.transform(test[feat].values.reshape(-1, 1)).squeeze()
        onehotinst = OneHotEncoder(categorical_features=onehots)
        train_final = onehotinst.fit_transform(train_encoded).toarray()
        test_final = onehotinst.transform(test_encoded).toarray()
        return train_final, test_final

    def _encode_y(self, train: pd.Series, test: pd.Series):
        """ Encode dependent variables """
        #NOTE: Assuming y is a single column
        assert len(train.columns) == 1
        [feature] = train.columns
        processor = self.preprocess_dict[feature]
        if isinstance(processor, list):
            processor_inst = processor[0]()
        else:
            processor_inst = processor()
        train_encoded = processor_inst.fit_transform(train.values.reshape(-1, 1))
        test_encoded = processor_inst.transform(test.values.reshape(-1, 1))
        if isinstance(processor, list):
            onehotinst = OneHotEncoder()
            train_final = onehotinst.fit_transform(train_encoded).toarray()
            test_final = onehotinst.transform(test_encoded).toarray()
        else:
            train_final, test_final = train_encoded.reshape(-1, 1), test_encoded.reshape(-1, 1)
        return train_final, test_final

    #NOTE: Scaling has to involve only the training set and that varies for cross-validation
    def encode(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
               target_label: str) -> EncodedData:
        """
            Encode the categorical features appropriately
            Parameters:
            ----------
            train_data : pd.DataFrame
                Training data
            test_data : pd.DataFrame
                Testing/Validation data
            target_label : str
                Label of the target/dependent column
            Returns:
            -------
            EncodedData - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                A tuple of (x_train, y_train, x_test, y_test)
        """
        assert all(train_data.columns == test_data.columns)
        features = list(train_data.columns)
        dep_features = [target_label]
        ind_features = [f for f in features if f not in dep_features]
        x_train, y_train = train_data[ind_features], train_data[dep_features]
        x_test, y_test = test_data[ind_features], test_data[dep_features]
        x_train_encoded, x_test_encoded = self._encode_x(x_train, x_test)
        y_train_encoded, y_test_encoded = self._encode_y(y_train, y_test)
        return x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded
