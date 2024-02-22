import logging

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


def main():
    # Examples: A pair of inputs/outputs used during training. 
    celsius_q    = numpy.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
    fahrenheit_a = numpy.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

    for i, c in enumerate(celsius_q):
        print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
    
    # Build the layers
    l0 = tensorflow.keras.layers.Dense(units=L0_N_NEURONS, input_shape=[1]) 

    # Assemble layers into the model
    model = tensorflow.keras.Sequential([l0])

    # Compile the model, with loss and optimizer functions
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER_FUNCTION)

    # Train the model
    history = model.fit(celsius_q, fahrenheit_a, epochs=EPOCHS, verbose=False)

    # Display tranining statistics
    pyplot.xlabel('Epoch Number')
    pyplot.ylabel("Loss Magnitude")
    pyplot.plot(history.history['loss'])
    pyplot.show()

    # Predict value
    result = model.predict([[100.0]])
    print(result)


if __name__ == '__main__':
    main()