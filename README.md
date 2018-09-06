# Deep-Reinforcement-Learning
Various Deep Reinforcement Learning techniques applied to different environments. Will be updated with new algorithms and environments often!


Just added (09/04/2018) is a Double Deep Q-learning (dqn) implementation applied to the Atari game: breakout! 

## Requirements

* python 3.6
* keras
* tensorflow
* gym
* gym['atari']
* numpy
* OpenCV
* random
* collections


## Getting Started

Running this code is very simple. Make sure that you have the above requirements taken care of, then download the two python files. In the command line, or any python editor change directory to where these two files are located and type:

`python dqn.py --train_dqn`

or for DDQN:

`python ddqn.py --train_ddqn`

If you can also render the atari environment by typing:

`python ddqn.py --train_ddqn --render`

To specify a certain number of episodes (example: 10) to run (50,000 default) use:

`python ddqn.py --train_ddqn --episodes 10`

Finally to test the final model after training, you can pass the test_dqn (or test_ddqn) flag:

`python ddqn.py --test_ddqn --render --episodes 10`


This code makes use of an atari wrapper code that can be found [here](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)

Priortized experience replay version of this code will be uploaded shortly!

**Recent Additions**:
* (09/04/2018) added DDQN for breakout
* (09/01/2018) added DQN for breakout


If you have any questions or comments, don't hesitate to send me an email! I am looking for ways to make this code even more computationally efficient. Contact info can be found [here](https://marcbrittain.github.io)


---
#### Support
If this helped you reduce time to develop, or if you found this work interesting feel free to send me a cup of coffee :)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/MarcBrittain)
