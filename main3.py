import logging
from typing import Any

import tensorflow
import numpy
from matplotlib import pyplot


logger = tensorflow.get_logger()
logger.setLevel(logging.ERROR)
                

L0_N_NEURONS = 1  # Number of neurons in the layer.

"""Loss function: A way of measuring how far off predictions are from the desired outcome.
The measured difference is called the 'loss'."""
LOSS_FUNCTION = 'mean_squared_error'

"""Learning rate: the step size taken when adjusting values in the model. 
If the value is too small, it will take too many iterations to train the model. 
Too large, and accuracy goes down. 
Finding a good value often involves some trial and error, 
but the range is usually within 0.001 (default), and 0.1"""
LEARNING_RATE = 0.1

"""Optimizer function: A way of adjusting internal values in order to reduce the loss."""
OPTIMIZER_FUNCTION = tensorflow.keras.optimizers.Adam(LEARNING_RATE)  

"""Epochs: How many times the cycle of calculate, compare, and adjust is run by the fit method."""
EPOCHS = 500


def create_nn_model() -> Any:
    """Create a neural network model."""
    # Build the layers
    l0 = tensorflow.keras.layers.Dense(units=L0_N_NEURONS, input_shape=[1]) 

    # Assemble layers into the model
    model = tensorflow.keras.Sequential([l0])

    # Compile the model, with loss and optimizer functions
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER_FUNCTION)

    return model


def train_nn_model(model: Any, 
                   input: numpy.ndarray[Any], 
                   output: numpy.ndarray[Any]) -> Any:
    """Train the neural network model and returns a history object."""
    history = model.fit(input, output, epochs=EPOCHS, verbose=False)
    return history


def display_training_statistis(history: Any) -> None:
    pyplot.xlabel('Epoch Number')
    pyplot.ylabel("Loss Magnitude")
    pyplot.plot(history.history['loss'])


def predict_value(model: Any, input: numpy.ndarray[Any]) -> numpy.ndarray[Any]:
    return model.predict(input)


def main():
    # Examples: A pair of inputs/outputs used during training. 
    fms_xor = [
        [[1], [1, -2], [1 -3], [2, 3, -1], [-2 -3]],
        [[1], [1, -2], [1 -3], [1, -4], [2, 3, 4, -1], [-2, -3], [-2, -4], [-3, -4]],
        [[1], [1, -2], [1 -3], [1, -4], [1, -5], [2, 3, 4, 5, -1], [-2, -3], [-2, -4], [-2, -5], [-3, -4], [-3, -5], [-4, -5]],
    ]
    fms_configs = [[2], [3], [4]]

    print(f'#Max: {max(len(clause) for model in fms_xor for clause in model)}')

    input = numpy.array(fms_xor,  dtype=int)
    output = numpy.array(fms_configs,  dtype=int)
    
    # Build the layers
    #l0 = tensorflow.keras.layers.Dense(units=L0_N_NEURONS, input_shape=(2,)) 

    # Assemble layers into the model
    #model = tensorflow.keras.Sequential([l0])

    # Compile the model, with loss and optimizer functions
    #model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER_FUNCTION)

    # Train the model
    #history = model.fit(input, output, epochs=EPOCHS, verbose=False)

    # Predict value
    #result = model.predict(input)

    #print(result)


if __name__ == '__main__':
    main()