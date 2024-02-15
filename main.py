import logging

import tensorflow
import numpy
from matplotlib import pyplot


logger = tensorflow.get_logger()
logger.setLevel(logging.ERROR)
                


def main():
    celsius_q    = numpy.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
    fahrenheit_a = numpy.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

    for i,c in enumerate(celsius_q):
        print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))


    l0 = tensorflow.keras.layers.Dense(units=1, input_shape=[1]) 
    model = tensorflow.keras.Sequential([l0])
    model.compile(loss='mean_squared_error', optimizer=tensorflow.keras.optimizers.Adam(0.1))
    history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
    print("Finished training the model")
    model.predict([100.0])
    print(model.predict([100.0]))
         
    pyplot.xlabel('Epoch Number')
    pyplot.ylabel("Loss Magnitude")
    pyplot.plot(history.history['loss'])


if __name__ == '__main__':
    main()