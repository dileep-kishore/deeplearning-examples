# @Author: dileep
# @Last Modified by:   dileep

from halo import Halo
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
        out = self.layer3(z)
        return out

class ChurnTrainer:
    """
        Trains the neural network on the churn dataset
        Parameters:
        ----------
        train_data : np.ndarray
            Training data for the model
        test_data : np.ndarray
            Testing data for the model
        #TODO: Define these later
        loss : str, optional
            Loss function to be used
        optimizer : str, optional
            Optimizer to be used
        learning_rate : float, optional
            Learning rate to be used in the optimizer
        Attributes:
        ----------
        model : nn.Module
            Pytorch neural network
    """
    def __init__(self, train_features: np.ndarray, train_labels: np.ndarray,
                 test_features: np.ndarray, test_labels: np.ndarray) -> None:
        self.train_data = TensorDataset(self._tofeat(train_features), self._tolabel(train_labels))
        self.test_data = TensorDataset(self._tofeat(test_features), self._tolabel(test_labels))
        self.input_size = train_features.shape[1]
        self.output_size = train_labels.shape[1]
        self.model = Classifier(self.input_size, 2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    @staticmethod
    def _tofeat(x):
        return torch.from_numpy(x).float()

    @staticmethod
    def _tolabel(x):
        return torch.from_numpy(x).long().squeeze()

    def train(self, epochs: int) -> float:
        """
            Trains the model and returns the accuracy on the test_data
            Parameters:
            ----------
            epochs : int
                Number of epochs for the training
            Returns:
            -------
            float
                Accuracy of the model on the test_data
        """
        trainloader = DataLoader(self.train_data, batch_size=100, shuffle=True, num_workers=2)
        testloader = DataLoader(self.test_data, batch_size=100, shuffle=True, num_workers=2)
        halo = Halo(text='Loading', spinner='dots')
        halo.start()
        for epoch in range(epochs):
            for i, data in enumerate(trainloader, 0):
                features, targets = data
                features = Variable(features, requires_grad=False)
                targets = Variable(targets, requires_grad=False)
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                halo.text = f"Epoch:{epoch}, Step:{(i+1)/40*100}, Loss:{loss.data[0]}"
        halo.stop()
        features = self.test_data.data_tensor
        targets = self.test_data.target_tensor
        features = Variable(features, requires_grad=False)
        _, output = self.model(features).max(dim=1)
        print(confusion_matrix(targets.numpy(), output.data.numpy()))
        print("accuracy", accuracy_score(targets.numpy(), output.data.numpy()))
