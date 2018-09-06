import os
import numpy as np
import random
import cv2
import time
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Flatten,Activation,LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from atari_wrapper import make_wrap_atari


### creating the DDQN Convolutional neural network for breakout


################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################


# initalize the DDQN agent
class DDQN_Agent:
    def __init__(self,state_size,action_size):

        self.memory = deque(maxlen=500000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # initial epsilon value


        # flag is for running an episode with epsilon = 1e-10
        self.flag = False

        # saving our values from the flag episode
        self.model_check = []

        self.learning_rate = 0.0001      #optimizer leanring rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.TRAIN_START = 5000  # how many samples to populate the replay memory with
        self.UPDATE_FREQ = 1000  # how often to update the target network

        self.count = 0


    def _build_model(self):


        # Consturct model
        model = Sequential()
        model.add(Convolution2D(32, (8, 8),input_shape=(video_width, video_height,stack_images), strides=(4, 4),activation='relu',padding='same'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu',padding='same'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu',padding='same'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(Dense(4,activation='linear'))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):

        """Grab samples from batch to train the network"""

        # grab the samples from memory
        minibatch = random.sample(self.memory, batch_size)
        for state,action,reward,next_state,done in minibatch:
            target = self.model.predict(state)


            if done:
                target[0][action] = reward
            else:
                # This is the heart of ddqn, we need to get the best actions
                # from the online model
                best_action = np.argmax(self.model.predict(next_state)[0])
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[best_action]

            self.model.fit(state, target, epochs=1, verbose=0)



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


    # action implementation for the agent
    def act(self, state):


        # simple epsilon-greedy strategy for the agent
        if random.random() <= self.epsilon:

            a = random.randrange(self.action_size)

        else:
            #act_values = self.model.predict(state)
            a = np.argmax(self.model.predict(state)[0])  # returns action

        # keep track of how many actions we have taken
        if not self.flag:
            self.count += 1

        return a



    def run_episode(self,n,init=False):
        """run the agent on breakout"""

        # initialize total reward for one trial
        total_reward = 0
        done = False

        s = env.reset()
        s = np.array(s)
        s = np.reshape(s,(1,video_height,video_width,4))


        while not done:

            # uncomment if you want to render
            #if not init:
            #    env.render()

            # ask our agent (nicely) what action we should take
            a = self.act(s)

            # take 1 discrete step in then environment
            next_s,reward,done,_ = env.step(a)

            # I have to do this np.array() step because of the atari_wrapper
            next_s = np.array(next_s)
            next_s = np.reshape(next_s,(1,video_height,video_width,4))


            # save this transition in memory
            self.remember(s,a,reward,next_s,done)


            # if we are not populating the replay or evaluating the model,
            # then train the model
            if not init and not self.flag and self.count % 4 == 0:
                self.replay(batch_size)


            # update the target model
            if self.count % self.UPDATE_FREQ == 0 and not init and not self.flag:
                self.update_target_model()

            # this is to stop populating the replay in the beginning...
            if self.count > self.TRAIN_START and init:
                return

            # update our total reward and update our current state: s
            total_reward += reward
            s = next_s


        # here is where I am saving stats from the evaluated model.
        # model evaluation happens when flag==True
        if self.flag:
            if len(self.model_check) > 100:
                p_avg = np.mean(self.model_check[-100:])
                self.model_check.append(total_reward)
                n_avg = np.mean(self.model_check[-100:])

                if n_avg > p_avg:
                    self.model.save_weights("ddqn_weights_opt.h5")


            else:
                self.model_check.append(total_reward)

            print("Model Check Complete, Episode {}, Score: {}".format(n,total_reward))



    def run_experient(self,EPISODES):
        """run experiment for the DDQN agent on breakout"""

        # epsilon values to be used during training
        epsilon_vals = np.linspace(self.epsilon,0.01,EPISODES)

        print("Initialzing Replay Memory with {} samples".format(self.TRAIN_START))
        print(" ")
        print("--------------------")
        while self.count < self.TRAIN_START:
            self.run_episode(0,init=True)


        print("--------------------")
        print(" ")
        print("training started....")
        print(" ")
        print("--------------------")
        self.count = 0
        for i in range(EPISODES):


            agent.epsilon = epsilon_vals[i]
            self.run_episode(i)


            # I do not want to evaluate the model at episode 0
            if i % model_check == 0 and i != 0:

                agent.flag = True
                agent.epsilon = 1e-10
                print(" ")
                print("--------------")
                print(" ")
                agent.run_episode(i)
                print(" ")
                print("--------------")
                print(" ")
                agent.flag = False







### I know globals are bad, but I think I can get away with a few!
env = make_wrap_atari('BreakoutNoFrameskip-v4')
EPISODES = 50000
batch_size = 32
model_check = 10
EPSILON_START = 1.0
video_width = 84
video_height = 84
stack_images = 4
agent = DDQN_Agent(env.observation_space.shape,env.action_space.n)
agent.epsilon = 1.0

# uncomment if you already have the weights saved
#agent.load('ddqn_weights_opt.h5')
#agent.update_target_model()
agent.run_experient(EPISODES)
