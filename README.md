## About IcyAI

Hello Internet! Welcome to *icyAI* - a project for which I recreated the game Icy Tower in Python with Pygame and built an AI that learns how to play it - here you can do so to.

[![Thumbnail](media/thumb6.png)](https://youtu.be/W6qyRbmr_aA)

Click [here](https://youtu.be/W6qyRbmr_aA), to see what I did in this project on YouTube.

Download the full project on [itch.io](https://nikp06.itch.io/icyai-icy-tower-ai-vs-human) (only Windows for now/I would appreciate someone making a mac or linux build and sending it to me :P -> I used pyinstaller on my windows machine).

I made use of a genetic algorithm called NEAT. NEAT evolves neural network topologies through neuroevolution.
It is a known method from the domain of reinforcement learning. The concept is further explained in the video. You can also read the initial [NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf) or browse through the [NEAT documentation](https://neat-python.readthedocs.io/en/latest/neat_overview.html).
This repository contains all files needed to train the AI for yourself.

![AI during training](media/training.gif)

## Description

This repository contains everything you need to play the game for yourself or to train your own Icy Tower AI.
Feel free to play around with the configuration file. Maybe you'll find a way to make the AI learn even more complex behavior. I'd be curious to know about it in case you do.

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
