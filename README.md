## About subwAI

*subwAI* - a project for training an AI to play the endless runner Subway Surfers using a supervised machine learning approach by imitation and a convolutional neural network (CNN) for image classification.

For this project, I made use of a supervised machine learning approach. I provided the ground truth data by playing the game and saving images with the corresponding action that was taken during the respective frame (jump, roll, left, right, noop) and in order for the AI to best imitate my playing style I used a convolutional neural network (CNN) with several layers (convolution, average pooling, dense layer, dropout, output), which gave me a good accuracy of 85% for it's predictions. After augmenting the data (mirroring, which resulted in a dataset twice as big) the model seemed to give even more robust results, when letting it play the game. Ultimately the model managed to finish runs of over a minute regularly and it safely handles the usual obstacles seen in the game. Moreover, the AI - with it's unconvential behavior - discovered a game-changing glitch.

More on all this can be seen in my [video](https://youtu.be/ZVSmPikcIP4) on YouTube.

[![new_thumb](https://user-images.githubusercontent.com/64498892/139440409-d6414a6e-2294-485e-bc36-b63ed623c8c2.png)](https://youtu.be/ZVSmPikcIP4)

## Description/Usage

This repository contains everything that is needed for building an AI that plays Subway Surfers.
With the provided scripts you can...
- build a dataset by playing the game while running ``` py ai.py gather ``` (takes rapid screenshots of the game and saves images in respective folders ['down', 'left', 'noop', 'right', 'up'] in the folder 'images'); press 'q' or 'esc' to quit
- train the specified model defined in get_model() on existing dataset running ``` py ai.py train ```; add ``` load <image_width> ``` to use a preloaded dataset for the respective image_width provided it has been saved before
- augment the existing dataset by flipping every image and adjust the label (flipped image in 'left' needs to be changed to 'right') by running ``` py dataset_augmentation.py ```
- have a look at what your trained model is doing under the hood with ``` py image_check.py ``` to see individual predictions for images and change labels when needed (press 'y' to move on to next image; 'n' to delete image; 'w' to move image to 'up'-folder; 'a' to move image to 'left'-folder; 's' to move image to 'down'-folder; 'd' to move image to 'right'-folder)
- if order of images is changed run ``` py image_sort.py ``` in order to bring everything in order again
- AND MOST IMPORTANTLY run ``` py ai.py play ``` to let the trained model play the game; press 'q' or 'esc' to quit; press 'y' to save a screen recording after the run and 'n' to not save it; add ``` auto ``` as a command line argument to have the program automatically save recordings of runs longer than 40 seconds

Also...
- in the folder 'recordings' you can view the saved screen captures and see the predictions for each individual frame as well as the frame rate
- in the folder 'models' your trained models are saved; while the Sequential() model (convolutional neural network with layers defined in get_model()) gives the best results you can also try other more simplistic machine learning models such as [KNeighborsClassifier(n_neighbors=5), GaussianNB(), Perceptron()]
- visualizations of the CNN-architecture and details regarding layer configurations as well as the accuracy and loss of the model is saved in models\Sequential

![ezgif com-gif-maker](https://user-images.githubusercontent.com/64498892/133991005-83309bec-ec01-4ea2-9a0e-20ccb7af73a6.gif)

