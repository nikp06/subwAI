from game import Game
import os
import cv2
import numpy as np
import sys

from sklearn.model_selection import train_test_split
import tensorflow as tf

EPOCHS = 10
IMG_WIDTH = 120
IMG_HEIGHT = 120
NUM_CATEGORIES = 5
TEST_SIZE = 0.3

PATH_TO_IMAGES = 'images\\training'
ACTIONS2IDX = {
    'left': 0,
    'right': 1,
    'up': 2,
    'down': 3,
    'noop': 4
}


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: py ai.py gather/train/play")

    if sys.argv[-1] == 'train':
        train_on_data()
    elif sys.argv[-1] == 'gather':
        gather_training_data()
    elif sys.argv[-1] == 'play':
        let_ai_play()


def train_on_data():
    # Get image arrays and labels for all image files
    images, labels = load_data()

    # Split data into training and testing sets
    print(labels)
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    filename = 'model'
    model.save(filename)
    print(f"Model saved to {filename}.")


def gather_training_data():
    game = Game()
    game.start_game()
    frame = None

    while True:
        while game.game_active:
            key = game.listen()
            frame = game.get_next_state(key, last_frame=frame)
        game.check_game_state()


def let_ai_play():
    # TODO: how to have one actions perform nothing
    game = Game()
    game.model = tf.keras.models.load_model('model')
    print(game.model.summary())

    game.start_game()
    frame = None
    key = None

    while True:
        while game.game_active:
            frame = game.get_next_state(key, frame)
            predictions = game.get_prediction(frame)
            # print(predictions)
            score = tf.nn.softmax(predictions[0])  # get logits
            # if round(max(predictions[0]), 2) > 0.98:
            action = game.actions[np.argmax(score)]
            # else:
            #     action = None
            print(action)
            game.take_action(action)
            # print("This image most likely belongs to {} with a {:.2f} percent confidence.\n"
            #       .format(action, 100 * np.max(score)))

        game.check_game_state()


def load_data():
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for folder in os.listdir(PATH_TO_IMAGES):
        folder_path = os.path.join(PATH_TO_IMAGES, str(folder))
        for image in os.listdir(folder_path):
            # reading image as an array
            im = cv2.imread(os.path.join(folder_path, image), cv2.IMREAD_COLOR)
            # resizing image to make input comparable
            im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
            images.append(im)
            labels.append(ACTIONS2IDX[str(folder)])
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # make sequential model
    # passing it as input a list of all the layers that we want to add instead of just adding them one after the other
    model = tf.keras.models.Sequential([
        # normalizing layer
        # tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # 1. Convolution step (Convolutional layer with e.g. 32 filters using a 3x3 kernel)
        # -> applying nr of diff filters to original image to get different feature maps
        # -> each feature map might extract diff relevant feature of the image that might be important
        # -> we can train NN to learn what those filters/values
        # of filters inside kernel should be in order to minimize loss

        # 2. Max-pooling layer, using 2x2 pool size
        # -> takes feature maps and reduces their dimensions to get fewer inputs
        # -> also makes NN much more resilient to tiny differences
        # -> (pixels don't have to be in the exact same spot in order to be classified the same)

        # first two steps can be applied multiple times (e.g to learn different features each time)
        # e.g. first convolution and pooling to learn low level features such as edges, curves, shapes
        # second step to learn higher level features such as objects (eyes etc.)

        # 3. Flatten units
        # -> now we have values that we can flatten out and put into traditional NN
        # -> one input for each value in each of the feature maps

        # 4. Make architecture of traditional neural network
        # -> hidden layers for calculating various different features of values from feature maps
        # Add a hidden layer with dropout
        # Add an output layer with output units for all 43 different roadsigns

        # ########################################

        # 1. first layer is conv layer
        # -> learn 32 diff filters with 3x3 kernels each
        # -> input shape is this case is dimensions of the images (in banknotes we had 4 different inputs)
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        # 2. pooling layer
        # -> look at 3x3 regions of the image and extract max-value to reduce size of input
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        # additional step of convolution and pooling
        # tf.keras.layers.Conv2D(
        #     32, (3, 3), activation="sigmoid", input_shape=(14, 14, 32)
        # ),
        # tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        # 3. Flatten units
        tf.keras.layers.Flatten(),

        # 4. make traditional NN architecture
        # -> densely connected hidden layer with 128 units
        # -> a NN without hidden layers only makes sense for linearly separable datasets
        # -> multiple layers allow us to get more complex functions
        # (multiple decision boundaries (one learned by each node in hidden layer)
        # that each account for some unique feature of the dataset)
        # -> to prevent over-fitting adding dropout
        # (randomly and temporarily leave out half of the nodes from this hidden layer)
        # -> lastly add an output layer with output units for all 43 different roadsigns
        # -> softmax takes output and turns it into a probability distribution
        tf.keras.layers.Dense(128, activation="relu"),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # accuracy shows how many guesses of the training dataset were correct
    # accuracy hopefully improving by repeating the process of gradient descent
    # GRADIENT DESCENT:
    # (stochastic (not all datapoints but randomly one each time) / mini-batch (small random sample each time))
    # Start with a random choice of weights.
    # This is our naive starting place, where we don’t know how much we should weight each input.
    # Repeat:
    # Calculate the gradient based on all data points that will lead to decreasing loss.
    # Ultimately, the gradient is a vector (a sequence of numbers).
    # Update weights according to the gradient.
    # (algorithm for minimizing loss to more accurately predict output)
    # (loss tells how bad hypothesis fct happens to be)
    # (telling in which direction we should be moving weights in order to minimize loss)
    # (learning not only weights but also the features/kernels to use)
    # BACKPROPAGATION:
    # Calculate error for output layer
    # For each layer, starting with output layer and moving inwards towards earliest hidden layer:
    # Propagate error back one layer. In other words, the current layer that’s being considered
    # sends the errors to the preceding layer.
    # Update weights.
    # (By knowing the error/loss of the output we can track backwards,
    # what nodes/weights and how much they were responsible for the error/loss
    # in order to know how to update weights)
    # (--> this is what makes NNs possible)
    # (taking multilevel structures and training them depending on
    # what values of weights are in order to figure out how to update weights to arrive at fct that minimizes loss best)

    # last output is from testing set

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()