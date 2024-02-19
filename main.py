import logging
from typing import Any

import tensorflow
import numpy
from matplotlib import pyplot

from nns4fms.models import FMInputCodification
from nns4fms.utils import utils


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


INPUT_DIR = 'fm_models/dimacs'
MAX_VARIABLE = 44079
MAX_CLAUSES = 100627
MAX_TERMS = 27


def main():
    # Examples: A pair of inputs/outputs used during training.
    dataset = [FMInputCodification(path) for path in utils.get_filepaths(INPUT_DIR, ['.dimacs'])]
    #max_terms = max(fm.max_clauses() for fm in dataset)
    #max_variables = max(fm.max_variable() for fm in dataset)
    #max_clauses = max(len(fm.clauses) for fm in dataset)
    inputs = []
    outputs = []
    for model in dataset:
        inputs.append(model.get_codification(MAX_TERMS, MAX_CLAUSES))
        outputs.append(model.get_configurations_number())
        

    nn_inputs = numpy.array(inputs, dtype=int)
    nn_outputs = numpy.array(outputs, dtype=int)

    # Build the layers
    input_layer = tensorflow.keras.layers.Flatten(input_shape=(MAX_CLAUSES, MAX_TERMS, 1)) 
    hidden_layer = tensorflow.keras.layers.Dense(units=128, activation=tensorflow.nn.relu)
    output_layer = tensorflow.keras.layers.Dense(units=1, activation=tensorflow.nn.softmax)

    # Assemble layers into the model
    model = tensorflow.keras.Sequential([input_layer, hidden_layer, output_layer])

    # Compile the model, with loss and optimizer functions
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(LEARNING_RATE),
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(nn_inputs, nn_outputs, epochs=EPOCHS, verbose=False)

    # Display training statistical
    pyplot.xlabel('Epoch Number')
    pyplot.ylabel("Loss Magnitude")
    pyplot.plot(history.history['loss'])
    
    # Predict value
    result = model.predict(input)
    print(f'Result: {result}')


if __name__ == '__main__':
    main()