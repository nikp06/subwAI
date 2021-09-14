## About subwAI

*subwAI* - a project for creating an AI that plays the endless runner Subway Surfers.

I made use of a supervised machine learning approach. I provided the ground truth data by playing the game and saving images with the corresponding action that was taken during the respective frame (jump, roll, left, right, noop) and in order for the AI to best imitate my playing style I used a convolutional neural network (CNN) with several layers (convolution, average pooling, dense layer, dropout, output), which gave me a good accuracy of 85% for it's predictions. After augmenting the data (mirroring, which resulted in a dataset twice as big) the model seemed to give even more robust results, when letting it play the game. Ultimately the model managed to many runs of over a minute and it safely handles the usual obstacles seen in the game. Moreover, the AI - with it's unconvential behavior - discovered a game-changing glitch.

More on all this can be seen in my [video](https://www.youtube.com/channel/UCV3IJuY11hfmjDomu6rEWTg) on YouTube.

## Description

This repository contains everything that is needed for building an AI that plays Subway Surfers.
With the provided scripts you can...
- build a dataset by playing the game while running ``` py ai.py gather ``` (takes rapid screenshots of the game and saves images in respective folders ['down', 'left', 'noop', 'right', 'up']); press 'q' or 'esc' to quit
- train the specified model defined in get_model() on existing dataset running ``` py ai.py train ```; add ``` load <image_width> ``` to use a preloaded dataset for the respective image_width provided it has been saved before
- augment the existing dataset by flipping every image and adjust the label (flipped image in 'left' needs to be changed to 'right') by running ``` py dataset_augmentation.py ```
- have a look at what your trained model is doing under the hood with ``` py image_check.py ``` to see individual predictions for images and change labels when needed (press 'y' to move on to next image; 'n' to delete image; 'w' to move image to 'up'-folder; 'a' to move image to 'left'-folder; 's' to move image to 'down'-folder; 'd' to move image to 'right'-folder)
- if order of images is changed run ``` py image_sort.py ``` in order to bring everything in order again
- AND MOST IMPORTANTLY run ``` py ai.py play ``` to let the trained model play the game; press 'q' or 'esc' to quit; press 'y' to save a screen recording after the run and 'n' to not save it; add ``` auto ``` as a command line argument to have the program automatically save recordings of runs longer than 40 seconds

Also...
- in the folder 'recordings' you can view the saved screen captures and see the predictions for each individual frame as well as the frame rate
- in the folder 'models' your trained models are saved; while the Sequential() model (convolutional neural network with layers defined in get_model()) gives the best results you can also try other more simplistic machine learning models such as [KNeighborsClassifier(n_neighbors=5), GaussianNB(), Perceptron()]
- visualizations of the CNN-architecture and details regarding layer configurations as well as the accuracy and loss of the model is saved in models\Sequential

[![Thumbnail](media/thumb6.png)](https://youtu.be/W6qyRbmr_aA)

![AI during training](media/training.gif)

## How to use

1. For starting the game:
```
py icyAI.py
```
2. Choose a screen-size (large is recommended, others might change physics of the game)

3. Simply navigate through the menu:
    * PLAY - play for yourself
    * TRAIN AI - train a new AI and specify how many generations
    * LET AI PLAY - choose a trained model and let the AI play and specify how many runs
    * HUMAN VS. AI - choose a trained model and play against this trained AI
![menu](media/menu2.png)
      
4. Enjoy the game and music, play around with the configuration file, experiment with parameters, analyze the statistics, speed up the simulation

Fitness Stats             |  Speciation Stats       |  Neural Network
:-------------------------:|:-------------------------:|:-------------------------:
![](media/TRAINING_PROCESS_avg_fitness50.png)  |  ![](media/TRAINING_PROCESS_speciation50.png)  |  ![](media/NN.png)

## Requirements and modules

- python 3
- pygame
- pickle
- neat
- sys
- os
- numpy
- tkinter
- random
- glob
- visualize
- re
- shutil
- time
