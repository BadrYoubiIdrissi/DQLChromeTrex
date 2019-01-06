#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import gym
import chrome_trex_gym

env = gym.make('chrome-trex-v0')


# In[2]:


WINDOW_LENGTH = 4
INPUT_SHAPE = (84,84)
nb_actions = 3

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# In[3]:


memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=10000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, 
               nb_steps_warmup=1000, gamma=.99, target_model_update=1000,
               train_interval=4, delta_clip=1., enable_dueling_network=True)
dqn.compile(Adam(lr=.000001), metrics=['mae'])


# In[ ]:


weights_filename = 'models/dqn_weights.h5f'
checkpoint_weights_filename = 'models/weights_{step}.h5f'
log_filename = 'dqn_log.json'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
callbacks += [FileLogger(log_filename, interval=100)]
dqn.fit(env, callbacks=callbacks, nb_steps=200000, log_interval=1000)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=10, visualize=False)
