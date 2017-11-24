# @Author: dileep
# @Last Modified by:   dileep

from halo import Halo
import pandas as pd
import torch.nn.functional as F
from torch import nn
from ...loaders import Churn

#TODO: Make this more flexible wrt hidden layers
class Classifier(nn.Module):
    """
        Pytorch classifier to classify data samples in the Churn Modelling dataset
        Parameters:
        ----------
        n_inputs : int
            Number of inputs/features used for classification
        n_outputs : int
            Number of classes that are part of the classification
        Attributes:
        ----------
        input_size : int
            Number of features the classifier takes as input
        output_size : int
            Number of classes the classifier needs to classify into
        hidden_size : Tuple
            Tuple containing integers that represent the size of each hidden layer
        n_layers : int
            Number of layer the classifier has
    """
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.input_size = n_inputs
        self.output_size = n_outputs
        self.n_layers = 3
        self.hidden_size = ((n_inputs + n_outputs) // 2, )
        self.layer1 = nn.Linear(n_inputs, self.hidden_size[0])
        self.layer2 = nn.Linear(self.hidden_size[0], self.hidden_size[0])
        self.layer3 = nn.Linear(self.hidden_size[0], n_outputs)

    def forward(self, x):
        """ Forward pass of the neural network """
        y = F.relu(self.layer1(x))
        z = F.relu(self.layer2(y))
        if self.output_size > 1:
            return F.softmax(self.layer3(z))
        else:
            return F.sigmoid(self.layer3(z))


