import logging
import math

import tensorflow
import tensorflow_datasets as tfds

import numpy
from matplotlib import pyplot


logger = tensorflow.get_logger()
logger.setLevel(logging.ERROR)

tfds.disable_progress_bar()


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


def normalize(images, labels):
  images = tensorflow.cast(images, tensorflow.float32)
  images /= 255
  return images, labels


def main():
    # Examples: A pair of inputs/outputs used during training. 
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    class_names = metadata.features['label'].names
    print("Class names: {}".format(class_names))
    
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    print("Number of training examples: {}".format(num_train_examples))
    print("Number of test examples:     {}".format(num_test_examples))

    # The map function applies the normalize function to each element in the train
    # and test datasets
    train_dataset =  train_dataset.map(normalize)
    test_dataset  =  test_dataset.map(normalize)

    # The first time you use the dataset, the images will be loaded from disk
    # Caching will keep them in memory, making training faster
    train_dataset =  train_dataset.cache()
    test_dataset  =  test_dataset.cache()

    # Take a single image, and remove the color dimension by reshaping
    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28,28))

    # Plot the image - voila a piece of fashion clothing
    pyplot.figure()
    pyplot.imshow(image, cmap=pyplot.cm.binary)
    pyplot.colorbar()
    pyplot.grid(False)
    pyplot.show()

    pyplot.figure(figsize=(10,10))
    for i, (image, label) in enumerate(train_dataset.take(25)):
        image = image.numpy().reshape((28,28))
        pyplot.subplot(5,5,i+1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.grid(False)
        pyplot.imshow(image, cmap=pyplot.cm.binary)
        pyplot.xlabel(class_names[label])
    pyplot.show()

    # Build the model
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu),
        tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax)
    ])

    # Compile the model, with loss and optimizer functions
    model.compile(optimizer='adam',
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    BATCH_SIZE = 32
    train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)
    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

    # Evaluate accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
    print('Accuracy on test dataset:', test_accuracy)

    # Make predictions and explore
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    print(predictions.shape)
    print(predictions[0])
    print(numpy.argmax(predictions[0]))
    print(test_labels[0])


if __name__ == '__main__':
    main()