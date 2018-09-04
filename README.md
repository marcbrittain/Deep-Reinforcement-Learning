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

```python
python dqn.py
```

or for DDQN:

```python
python ddqn.py
```


This code makes use of an atari wrapper code that can be found [here](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)

Priortized experience replay version of this code will be uploaded shortly!

Recent Modifications:
* (09/04/2018) added DDQN for breakout
* (09/01/2018) added DQN for breakout


If you have any questions or comments, don't hesitate to send me an email! Contact info can be found [here](https://marcbrittain.github.io)
