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
import argparse


### creating the DDQN Convolutional neural network for breakout


################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################



############################################################

# Original Code for Class: SumTree and ReplayMemory can be
# found here: https://github.com/jaara/AI-blog

############################################################



class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayMemory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def get_batch(self, n):
        batch = []
        ids = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            ids.append(idx)

        return ids,batch

    def update(self, idx, error):
        for i in range(len(idx)):
            i_d = idx[i]
            e = error[i]
            p = self._getPriority(e)

            self.tree.update(i_d, p)




# initalize the DDQN agent
class DDQN_Agent:
    def __init__(self,state_size,action_size):

        self.memory = ReplayMemory(500000)
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
        error = abs(reward)
        self.memory.add(error,(state, action, reward, next_state, done))


    def replay(self, batch_size):

        """Grab samples from batch to train the network"""

        # grab the samples from memory
        idx,minibatch = self.memory.get_batch(batch_size)

        # converting the minibatch into arrays of size:
        #       (batch, state), (batch, action), ...
        # Also converting the terminal states to integer

        idx = np.vstack(np.array(idx)).astype(int)[:,0]
        state = np.vstack(np.array(minibatch)[:,0]).astype(float)
        action = np.vstack(np.array(minibatch)[:,1]).astype(int)[:,0]
        reward = np.vstack(np.array(minibatch)[:,2]).astype(float)[:,0]
        next_state = np.vstack(np.array(minibatch)[:,3]).astype(float)
        done = np.vstack(np.array(minibatch)[:,4]).astype(int)[:,0]+0

        indices = np.arange(0,batch_size)

        # Q value from target network
        target_q_values = self.target_model.predict(next_state)


        target = self.model.predict(state)


        old_val = target[indices,action]


        # this is the heart of ddqn. We need to grab the best actions from the online model

        best_actions = np.argmax(self.model.predict(next_state),axis=1)


        # Now we can update all of the target values at once using numpy...
        # I think this should really help with batch scaling

        target[indices,action] = reward + (1-done)*self.gamma*target_q_values[indices,best_actions]
        error = np.abs(old_val-target[indices,action])
        self.memory.update(idx,error)
        # train the model with our states and targets
        self.model.fit(state,target,batch_size=batch_size,verbose=0,epochs=1)


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



    def run_episode(self,n,args,init=False):
        """run the agent on breakout"""

        # initialize total reward for one trial
        total_reward = 0
        done = False

        s = env.reset()
        s = np.array(s)
        s = np.reshape(s,(1,video_height,video_width,4))


        while not done:

            # uncomment if you want to render
            if not init and args.render:
                env.render()

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

                if n_avg > p_avg and args.train_ddqn:
                    self.model.save_weights("ddqn_per_weights.h5")


            else:
                self.model_check.append(total_reward)

            print("Model Check Complete, Episode {}, Score: {}".format(n,total_reward))


    def run_experient(self,args):
        """run experiment for the DDQN agent on breakout"""



        if args.test_ddqn:
            agent.load('ddqn_per_weights.h5')
            agent.update_target_model()
            agent.epsilon=1e-10
            self.flag = True

            print("--------------------")
            print(" ")
            print("testing started....")
            print(" ")
            print("--------------------")
            for i in range(args.episodes):
                agent.run_episode(i,args)


            return



        # epsilon values to be used during training
        epsilon_vals = np.linspace(self.epsilon,0.01,args.episodes)

        print("Initialzing Replay Memory with {} samples".format(self.TRAIN_START))
        print(" ")
        print("--------------------")
        while self.count < self.TRAIN_START:
            self.run_episode(0,args,init=True)


        print("--------------------")
        print(" ")
        print("training started....")
        print(" ")
        print("--------------------")
        self.count = 0
        for i in range(args.episodes):


            agent.epsilon = epsilon_vals[i]
            self.run_episode(i,args)


            # I do not want to evaluate the model at episode 0
            if i % model_check == 0 and i != 0:

                agent.flag = True
                agent.epsilon = 1e-10
                print(" ")
                print("--------------")
                print(" ")
                agent.run_episode(i,args)
                print(" ")
                print("--------------")
                print(" ")
                agent.flag = False




### parsing the input from the command line
#######################################################################################################
parser = argparse.ArgumentParser(description="DQN Breakout")
parser.add_argument('--train_ddqn', action='store_true', help='whether train DQN')
parser.add_argument('--test_ddqn', action='store_true', help='whether test DQN')
parser.add_argument('--render', action='store_true', help='whether render environment or not')
parser.add_argument('--episodes', type=int, default = 50000, help='Number of episodes to run')
args = parser.parse_args()
#######################################################################################################





### I know globals are bad, but I think I can get away with a few!
env = make_wrap_atari('BreakoutNoFrameskip-v4')
batch_size = 32
model_check = 10
EPSILON_START = 1.0
video_width = 84
video_height = 84
stack_images = 4
agent = DDQN_Agent(env.observation_space.shape,env.action_space.n)
agent.epsilon = 1.0
agent.run_experient(args)
