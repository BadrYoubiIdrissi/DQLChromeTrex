import matplotlib.pyplot as plt
from .Memory import Memory
from .model import get_models
import numpy as np
from keras.utils import to_categorical


### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
batch_size = 64             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000          # Number of experiences the Memory can keep

def prepopulateMemory(memory, env, pre_size):
    for i in range(pre_size):
        action = env.action_space.sample()
        if i == 0 or env.game.gameOver :
            if env.game.gameOver:
                env.reset()
            obs = env.getStackedObservation()
        new_obs, reward, done, _ = env.step(action)
        memory.add((obs,action,reward,new_obs, done))
        obs = new_obs

def epsilon_greedy(counter):
    return explore_stop + (explore_start-explore_stop)*np.exp(-counter*decay_rate)

def train(env):

    model, train_model = get_models()
    memory = Memory(memory_size)
    prepopulateMemory(memory, env, pretrain_length)
    decay_counter = 0

    for i in range(total_episodes):
        env.reset()
        obs, reward, done, _ = env.step(0)

        while not done:
            epsilon = epsilon_greedy(decay_counter)

            if decay_counter % 4 == 0:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    Q = model.predict(obs[np.newaxis,:,:,:])
                    action = np.argmax(Q)

            new_obs, reward, done, _ = env.step(action)
            
            memory.add((obs,action,reward,new_obs,done))

            obs = new_obs

            if decay_counter % 4 == 0:
                ##Learning

                batch = memory.sample(batch_size)

                batch_obs = np.array([entry[0] for entry in batch])
                batch_actions = to_categorical(np.array([entry[1] for entry in batch]))
                batch_rewards = np.array([entry[2] for entry in batch])
                batch_next_obs = np.array(np.array([entry[3] for entry in batch]))
                batch_not_dones = np.array(np.array([not entry[4] for entry in batch]), dtype=np.uint8)
                
                Q_next_obs = model.predict(batch_next_obs)
                y = batch_rewards + batch_not_dones*gamma*(Q_next_obs.max(axis=1))

                train_model.train_on_batch(x=[batch_obs, batch_actions],y=y)

            decay_counter += 1

        print(epsilon)
        if i % 5 == 0:
            model.save("models/model_episode_{}.h5".format(i))

