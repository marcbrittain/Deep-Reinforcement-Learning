import tensorflow as tf
import gym
import numpy as np
import random
import time
from collections import deque
from sklearn.preprocessing import normalize,MinMaxScaler


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)


### Created by      Marc Brittain
###            marcbrittain.github.io
###               mwb@iastate.edu



class Agent:
    def __init__(self,state_size,action_size,num_episodes,env):

        self.state_size = state_size
        self.action_size = action_size

        self.env = env
        self.transform = MinMaxScaler(feature_range=(-1,1))
        self.transform.fit([self.env.observation_space.high, self.env.observation_space.low])

        self.max = self.env.observation_space.high.reshape(-1,self.state_size)

        self.memory = deque(maxlen=50000)
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.batch_size = 128
        self.flag = False
        self.num_episodes = num_episodes
        self.model_update = 1000
        self.epsilon = 0.05
        #self.epsilon = np.linspace(1.0,0.001,num_episodes)
        self.count = 0
        self.eval_interval = 20
        self.best_reward = -100000

        self._type = 'dueling'
        self._double = True
        self.init = 50000

        self.model = self.build_NN()
        self.target_model = self.build_NN()





    def build_NN(self):


        if self._type == 'dueling':

            I = tf.keras.layers.Input(shape=(self.state_size,))

            Input = tf.keras.layers.Dense(64,activation='relu')(I)
            H1 = tf.keras.layers.Dense(64,activation='relu')(Input)
            x = tf.keras.layers.Dense(self.action_size+1,activation='linear')(H1)

            # output layer can be changed to tf.keras.backend.max instead of mean
            outputlayer = tf.keras.layers.Lambda(lambda a:
                         tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.mean(a[:, 1:],
                         axis=1,
                         keepdims=True),
                         output_shape=(self.action_size,))(x)

            model = tf.keras.models.Model(inputs=I, outputs=outputlayer)
            adam = tf.keras.optimizers.Adam(lr = self.learning_rate)
            model.compile(adam, loss='mse')

            return model

        else:

            I = tf.keras.layers.Input(shape=(self.state_size,))

            Input = tf.keras.layers.Dense(64,activation='relu')(I)
            H1 = tf.keras.layers.Dense(64,activation='relu')(Input)
            outputlayer = tf.keras.layers.Dense(self.action_size,activation='linear')(H1)
            model = tf.keras.models.Model(inputs=I, outputs=outputlayer)
            adam = tf.keras.optimizers.Adam(lr = self.learning_rate)
            model.compile(adam, loss='mse')

            return model


    def remember(self,s,a,r,sp,T):

        self.memory.append([s,a,r,sp,T])



    def train(self):

        transitions = random.sample(self.memory,self.batch_size)

        # this could probably be made more efficient
        state = np.array([rep[0] for rep in transitions]).reshape(self.batch_size,self.state_size)
        next_state  = np.array([rep[3] for rep in transitions]).reshape(self.batch_size,self.state_size)
        reward = np.array([rep[2] for rep in transitions])
        action = np.array([rep[1] for rep in transitions])
        done = np.array([rep[4] for rep in transitions])

        target = self.model.predict(state,self.batch_size)
        best_action = self.model.predict(next_state,self.batch_size)
        t = self.target_model.predict(next_state,self.batch_size)

        for i in range(self.batch_size):
            old_val = target[i][action[i]]
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self._double:
                    ba = np.argmax(best_action[i])
                    target[i][action[i]] = reward[i] + self.gamma * t[i][ba]
                else:
                    target[i][action[i]] = reward[i] + self.gamma * np.max(t[i])


        self.model.fit(state, target, batch_size=self.batch_size,epochs=1, verbose=0)


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self,s,e):

        if random.random() < e:
            return random.randrange(self.action_size)

        else:
            return np.argmax(self.model.predict(s,batch_size=1)[0])



    def init_memory(self):
        """Initialize replay memory"""
        count = 0
        while count < self.init:
            s = self.env.reset()
            s = self.transform.transform(s.reshape(1,self.state_size))

            done = False
            total_reward = 0
            while not done:
                a = self.act(s,1.0)

                sp,r,done,_ = self.env.step(a)
                sp = self.transform.transform(sp.reshape(1,self.state_size))

                count += 1
                self.remember(s,a,r,sp,done)

                s = sp

                if count == 50000:
                    break



    def run_experiment(self,training):
        self.training_scores = []
        self.evaluation_scores = []

        self.init_memory()
        if not training:
            self.model.load_weights('weights_best.h5')

        for i in range(self.num_episodes):
            if training:
            	#e = self.epsilon[i]
                e = self.epsilon

            else:
                e = 1e-10


            s = self.env.reset()
            s = self.transform.transform(s.reshape(1,self.state_size))

            done = False
            total_reward = 0
            while not done:
                a = self.act(s,e)

                sp,r,done,_ = self.env.step(a)
                sp = self.transform.transform(sp.reshape(1,self.state_size))

                self.count += 1
                total_reward += r


                if training:
                    self.remember(s,a,r,sp,done)
                    self.train()

                if training and self.count % self.model_update == 0:
                    self.update_target_model()

                s = sp

            self.training_scores.append(total_reward)


            # Now running an extra episode to check how the model is doing
            if i % self.eval_interval == 0 and training:
                s = self.env.reset()

                s = self.transform.transform(s.reshape(1,self.state_size))

                done = False
                total_reward = 0
                while not done:
                    a = self.act(s,1e-10)
                    sp,r,done,_ = self.env.step(a)
                    sp = self.transform.transform(sp.reshape(1,self.state_size))


                    total_reward += r
                    s = sp

                self.evaluation_scores.append(total_reward)

                if total_reward > self.best_reward:
                    self.model.save_weights('weights_best.h5')
                    self.best_reward = total_reward

                if len(self.evaluation_scores) > 10:

                    print('Episode: {} | Score: {} | Avg. Eval Score: {} | Avg. Train Score: {}'.format(i,total_reward,np.mean(self.evaluation_scores[-10:]),np.mean(self.training_scores[-100:])))

                else:
                    print('Episode: {} | Score: {}'.format(i,total_reward))





env = gym.make('CartPole-v0')
env._max_episode_steps = 200
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


agent  = Agent(state_size,action_size,1000,env)
start = time.time()
agent.run_experiment(True)
end = time.time()


agent.model.save_weights('weights.h5')
training_results = np.array(agent.training_scores)
eval_results = np.array(agent.evaluation_scores)
np.save('training_results.npy',training_results)
np.save('eval_training.npy',eval_results)
