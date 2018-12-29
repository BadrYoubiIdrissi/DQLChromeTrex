import matplotlib.pyplot as plt

def train(env, number_episodes=1):

    for i in range(number_episodes):
        action = 1
        observation, reward, done, info = env.step(action)
        while not done:
            action = 1
            observation, reward, done, info = env.step(action)
    for i in range(len(observation)):
        plt.imshow(observation[0])
        plt.show()