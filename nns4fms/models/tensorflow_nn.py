from abc import abstractmethod
from typing import Any

import tensorflow
import numpy
from matplotlib import pyplot

from nns4fms.models import NeuralNetwork


class TensorFlowNN(NeuralNetwork):
    """An implementation of a neural network model using TensorFlow and Keras."""

    # Default values
    LOSS_FUNCTION: str = 'mean_squared_error'
    OPTIMIZER_FUNCTION: Any = tensorflow.keras.optimizers.Adam(NeuralNetwork.LEARNING_RATE)  

    @abstractmethod
    def __init__(self, 
                 loss_function: Any = LOSS_FUNCTION, 
                 optimizer_function: Any = OPTIMIZER_FUNCTION, 
                 learning_rate: float = NeuralNetwork.LEARNING_RATE) -> None:
        self._model = self._create_model()
        self._model.compile(loss=loss_function, optimizer=optimizer_function)

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the neural network model building and assembling the layer into the model."""
        pass

    def train(self, examples: tuple[list[Any], list[Any]], epochs: int = NeuralNetwork.EPOCHS) -> Any:
        input, output = examples
        self._history = self._model.fit(input, output, epochs=epochs, verbose=False)
        return self._model

    @abstractmethod
    def predict(self, input: Any) -> Any:
        return self._model.predict(input)

    @abstractmethod
    def display_training_statistis(self) -> Any:
        pyplot.xlabel('Epoch Number')
        pyplot.ylabel("Loss Magnitude")
        pyplot.plot(self._history.history['loss'])
        return None
