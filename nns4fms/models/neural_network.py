from abc import ABC, abstractmethod
from typing import Any


class NeuralNetwork(ABC):
    """A generic neural network model."""

    # Default values
    EPOCHS: int = 500
    LEARNING_RATE: float = 0.001

    @abstractmethod
    def __init__(self, 
                 loss_function: Any, 
                 optimizer_function: Any, 
                 learning_rate: float = LEARNING_RATE) -> None:
        """Create a neural network model.
        
        - loss_function: A way of measuring how far off predictions are from the desired outcome.
            The measured difference is called the "loss".
            Example. Mean squared error (MSE), a type of loss function that counts a small number 
                of large discrepancies as worse than a large number of small ones.

        - optimizer_function: A specific implementation of the gradient descent algorithm.
            There are many algorithms for this. 
            The "Adam" Optimizer, which stands for ADAptive with Momentum, 
            is considered the best-practice optimizer.

        - learning_rate:  The "step size" for loss improvement during gradient descent.
            If the value is too small, it will take too many iterations to train the model. 
            Too large, and accuracy goes down. 
            Finding a good value often involves some trial and error, 
            but the range is usually within 0.001 (default), and 0.1
        """
        pass

    @abstractmethod
    def train(self, examples: tuple[list[Any], list[Any]], epochs: int = EPOCHS) -> Any:
        """Train the neural network model with the examples for a givem number of epochs.

        - examples: a tuple of lists with the inputs/outputs pairs used for training.
        - epochs: number of full passes over the entire training dataset.
        """

    @abstractmethod
    def predict(self, input: Any) -> Any:
        """Make a prediction using the given input."""
        pass

    @abstractmethod
    def display_training_statistis(self) -> Any:
        """Display training statistics such as plotting how the loss of the model goes down
            after each training epoch."""
        pass
