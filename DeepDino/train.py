import matplotlib.pyplot as plt

def train(env, number_episodes=1):

    for i in range(number_episodes):
        action = 1
        observation, reward, done, info = env.step(action)
        while not done:
            action = 1
            observation, reward, done, info = env.step(action)
    plt.imshow(observation.T)
    plt.show()