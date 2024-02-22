import logging
from typing import Any

from alive_progress import alive_bar, alive_it

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
EPOCHS = 100


INPUT_DIR = 'generated/'
FM_TO_PREDICT = 'fm_models/dimacs/Pizzas.dimacs'
MAX_VARIABLE = 12  # 44079
MAX_CLAUSES = 51  # 100627
MAX_TERMS = 10  # 27


def main():
    # Examples: A pair of inputs/outputs used during training.
    dataset = [FMInputCodification(path) for path in 
               alive_it(utils.get_filepaths(INPUT_DIR, ['.dimacs'])[:50000], 
                        title=f'Getting dataset from {INPUT_DIR}...')]
    #max_terms = max(fm.max_clauses() for fm in dataset)
    #max_variables = max(fm.max_variable() for fm in dataset)
    #max_clauses = max(len(fm.clauses) for fm in dataset)
    inputs = []
    outputs = []
    with alive_bar(len(dataset)) as bar:
        bar.title('Codifying inputs/outputs...')
        for model in dataset:
            inputs.append(model.get_codification(MAX_TERMS, MAX_CLAUSES))
            outputs.append(model.get_configurations_number())
            bar()
        
    nn_inputs = numpy.array(inputs, dtype=int)
    nn_outputs = numpy.array(outputs, dtype=int)
    print(f'Inputs: {nn_inputs}')
    print(f'Outputs: {nn_outputs}')
    print(nn_inputs[0])

    # Assemble layers into the model
    print(f'Assembling the layers...')
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=(MAX_CLAUSES, MAX_TERMS, 1)),
        tensorflow.keras.layers.Dense(units=51, activation=tensorflow.nn.relu),
        tensorflow.keras.layers.Dense(units=10, activation=tensorflow.nn.sigmoid),
        #tensorflow.keras.layers.Dense(units=12, activation=tensorflow.nn.relu),
        tensorflow.keras.layers.Dense(units=1)
    ])

    # Compile the model, with loss and optimizer functions
    print(f'Compiling the model...')
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(LEARNING_RATE),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Train the model
    print(f'Training the model...')
    history = model.fit(nn_inputs, nn_outputs, epochs=EPOCHS, verbose=False)

    # Display training statistical
    print(f'Displaying training statistical...')
    pyplot.xlabel('Epoch Number')
    pyplot.ylabel("Loss Magnitude")
    pyplot.plot(history.history['loss'])
    pyplot.show()

    # evaluate the keras model
    _, accuracy = model.evaluate(nn_inputs, nn_outputs)
    print('Accuracy: %.2f' % (accuracy*100))

    # Predict value
    print(f'Predicting value for {FM_TO_PREDICT}...')
    fm_to_predict = FMInputCodification(FM_TO_PREDICT)
    nn_input = numpy.array([fm_to_predict.get_codification(MAX_TERMS, MAX_CLAUSES)], dtype=int)
    result = model.predict(nn_input)
    print(f'Result: {result}')


if __name__ == '__main__':
    main()