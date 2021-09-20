from game import Game
import os
import time
import cv2
import numpy as np
import sys
from joblib import dump, load
import visualkeras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf

EPOCHS = 20
if len(sys.argv) > 3:
    IMG_WIDTH = int(sys.argv[-1])
    IMG_HEIGHT = int(sys.argv[-1])
else:
    IMG_WIDTH = 96
    IMG_HEIGHT = 96
NUM_CATEGORIES = 5
TEST_SIZE = 0.3

PATH_TO_IMAGES = 'images\\training'
ACTIONS2IDX = {
    'left': 0,
    'right': 1,
    'up': 2,
    'down': 3,
    'noop': 4,
    'left_flipped': 1,
    'right_flipped': 0,
    'up_flipped': 2,
    'down_flipped': 3,
    'noop_flipped': 4
}


def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        sys.exit("Usage: py ai.py gather/train/play")

    if sys.argv[1] == 'train':
        train_on_data()
    elif sys.argv[1] == 'gather':
        gather_training_data()
    elif sys.argv[1] == 'play':
        let_ai_play()


def train_on_data():
    """
    Trains a convolutional neural networks (or others if specified in models) according to layer-specification
    in get_model().
    """
    pixels, images, labels = load_data()

    labels = tf.keras.utils.to_categorical(labels)
    # Split data into training and testing sets
    labels_nr = np.argmax(labels, axis=1).reshape((labels.shape[0], ))

    # models = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), Perceptron(), tf.keras.models.Sequential()]
    models = [tf.keras.models.Sequential()]
    accuracies = []
    for classifier in models:
        filename = os.path.join('models', type(classifier).__name__)
        if type(classifier).__name__ == 'Sequential':
            # Get a compiled neural network
            model = get_model()
            visualkeras.layered_view(model, to_file=os.path.join('models', 'Sequential', 'architecture.png')).show()

            # Split data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=TEST_SIZE)

            # Fit model on training data
            model.fit(x_train, y_train, epochs=EPOCHS)

            # Evaluate neural network performance
            metrics = model.evaluate(x_test,  y_test, verbose=2)

            # Make predictions on the testing set
            predictions = model.predict(x_test)
            predictions = np.argmax(predictions, axis=1).reshape((predictions.shape[0], ))
            y_test = np.argmax(y_test, axis=1).reshape((y_test.shape[0], ))

            # Save model to file
            model.save(filename)
            print(f"Model saved to {filename}.")
            print(model.summary())

            # log to text file
            with open(os.path.join('models', 'report.txt'), 'a') as fh:
                # Pass the file handle in as a lambda function to make it callable
                model.summary(line_length=100, print_fn=lambda x: fh.write(x + '\n'))
                fh.write('Epochs: ' + str(EPOCHS) + '\n')
                fh.write('Accuracy: ' + str(metrics[1]) + '% | Loss: ' + str(metrics[0]) + '\n\n')
                for i in range(7):
                    try:
                        layer_config = model.get_layer(index=i).get_config()
                        fh.write('Layer ' + str(i) + ' config: ' + str(layer_config)+'\n')
                    except:
                        break
                fh.write('\n')
            model_path = os.path.join('models', 'Sequential', 'model_'+str(round(metrics[1], 2)*100)+'.png')
            tf.keras.utils.plot_model(model, to_file=model_path, show_shapes=True, show_layer_names=True)

        else:
            model = classifier

            x_train, x_test, y_train, y_test = train_test_split(pixels, np.array(labels_nr), test_size=TEST_SIZE)

            # Fit model on training data
            model.fit(x_train, y_train)
            
            # Make predictions on the testing set
            predictions = model.predict(x_test)

            # save model
            dump(model, filename + '.joblib')
            print(f"Model saved to {filename}.")

        # Compute how well we performed
        correct = (y_test == predictions).sum()
        incorrect = (y_test != predictions).sum()
        total = len(predictions)

        # Print results
        print(f"Results for model {type(model).__name__}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {100 * correct / total:.2f}%")

        accuracies.append(100 * correct / total)

    print(accuracies)


def gather_training_data():
    """
    Starts playing loop and saves individual frames into respective folders depending on actions taken.
    """
    game = Game()
    game.disable_wifi()
    game.start_game()
    frame = None
    counter = 0
    while True:
        # start_time = time.time()
        while game.game_active:
            if game.intro:
                game.intro = (game.last_time - game.game_start) < 1.5

            counter += 1
            key = game.listen()
            frame = game.get_next_state(key, last_frame=frame)
            game.timer()
        game.check_game_state()


def let_ai_play():
    """
    Loads saved model and starts playing loop with model making predictions for individual frames.
    """
    game = Game()
    game.disable_wifi()
    # [KNeighborsClassifier(n_neighbors=5), tf.keras.models.Sequential(), GaussianNB(), Perceptron()]
    model = tf.keras.models.Sequential()
    path = os.path.join('models', type(model).__name__+'_whole_set')

    if type(model).__name__ == 'Sequential':
        game.NN = True
        game.model = tf.keras.models.load_model(path)
        print(game.model.summary())
    else:
        game.model = load(path + '.joblib')
        game.NN = False

    game.start_game()
    frame = None
    key = None
    last_action = None
    jump = False
    jump_start = 0

    while True:
        while game.game_active:
            if game.intro:
                game.intro = (game.last_time - game.game_start) < 1.5

            frame = game.get_next_state(key, frame)
            # start_time = time.time()
            predictions, cap = game.get_prediction(frame)

            if game.NN:
                score = tf.nn.softmax(predictions[0])  # get logits
                action = game.actions[np.argmax(predictions)]   # if np.amax(predictions) > 0.7 else 'noop'
            else:
                action = game.actions[predictions[0]]

            # if round(max(predictions[0]), 2) > 0.98:

            if last_action != action:
                if action == 'down' and jump:
                    action = 'noop'
                elif action == 'left' and last_action == 'right' or action == 'right' and last_action == 'left':
                    action = 'noop'
                else:
                    game.take_action(action)
                if action == 'up' and jump is False:
                    jump = True
                    jump_start = time.time()
                if jump and (game.last_time-jump_start) > 1:
                    jump = False
                    
                # frame rate
                # print("--- %s seconds ---" % (time.time() - start_time))
                
            print("{} with {:.2f} percent certainty" .format(action, 100 * np.max(score)))
            last_action = action if action == 'left' or action == 'right' else 'noop'

            # print("This image most likely belongs to {} with a {:.2f} percent confidence.\n"
            #       .format(action, 100 * np.max(score)))

            game.screen_cap(cap, action)
            game.last_time = time.time()
            
            game.timer()

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
    if len(sys.argv) > 2:
        if sys.argv[2] == 'load':
            images = np.load(os.path.join('dataset_preloaded', 'images'+str(IMG_WIDTH)+'.npy'))
            labels = np.load(os.path.join('dataset_preloaded', 'labels'+str(IMG_WIDTH)+'.npy'))
            pixels = np.load(os.path.join('dataset_preloaded', 'pixels'+str(IMG_WIDTH)+'.npy'))

            return pixels, images, labels
    else:
        images = []
        labels = []

        counts = [0, 0, 0, 0, 0]
        
        folders = ['down', 'down_flipped', 'up', 'up_flipped', 'left', 'right_flipped',
                   'right', 'left_flipped', 'noop', 'noop_flipped']
        
        min_images = len(os.listdir(os.path.join(PATH_TO_IMAGES, 'down')))*2
        
        for folder in folders:
            folder_path = os.path.join(PATH_TO_IMAGES, str(folder))
            
            counter = 0
            print(folder)
            for image in os.listdir(folder_path):
                # for equally sized classes (same amount as down class which has least images
                if counts[ACTIONS2IDX[folder]] == min_images:
                    print("limit reached")
                    break
                counts[ACTIONS2IDX[folder]] += 1
                counter += 1
                # reading image as an array
                im = cv2.imread(os.path.join(folder_path, image), cv2.IMREAD_COLOR)
                # resizing image to make input comparable
                im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                images.append(im)
                labels.append(ACTIONS2IDX[str(folder)])
        
        print(counts)
        
        # for image in images:
        x_data = np.array([np.array(image) for image in images])

        pixels = x_data.flatten().reshape(int(len(images)), int(x_data.size/len(images)))

        np.save(os.path.join('dataset_preloaded', 'images'+str(IMG_WIDTH)+'.npy'), np.array(images))
        np.save(os.path.join('dataset_preloaded', 'labels'+str(IMG_WIDTH)+'.npy'), np.array(labels))
        np.save(os.path.join('dataset_preloaded', 'pixels'+str(IMG_WIDTH)+'.npy'), np.array(pixels))

        return pixels, images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # TF IMPLEMENTATION
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
            32, (3, 3), activation="tanh", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        # 2. pooling layer
        # -> look at 3x3 regions of the image and extract max-value to reduce size of input
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        # additional step of convolution and pooling
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="tanh"  # input_shape=(31, 31, 32)
        ),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

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
        # tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Dense(128, activation="relu"),
        # tf.keras.layers.Dropout(0.3),

        # has to be softmax apparently
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # KERAS IMPLEMENTATION
    # input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    # img_input = k.Input(shape=input_shape)
    # conv1 = layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=input_shape)(img_input)
    # pool1 = layers.AveragePooling2D(pool_size=(3, 3))(conv1)
    # conv2 = layers.Conv2D(16, (2, 2), activation='sigmoid')(pool1)
    # pool2 = layers.AveragePooling2D(pool_size=(2, 2))(conv2)
    # flat1 = layers.Flatten()(pool2)
    # dense1 = layers.Dense(128, activation='relu')(flat1)
    # dense2 = layers.Dense(NUM_CATEGORIES, activation='softmax')(dense1)
    #
    # model = models.Model(img_input, dense2)

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
