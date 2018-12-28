import argparse
import gym
import gym_chrome_dino
import matplotlib.pyplot as plt

def main(args):
    if not args.test:
        if not args.headless:
            env = gym.make("ChromeDino-v0")
        else:
            env = gym.make("ChromeDinoNoBrowser-v0")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
        plt.imshow(observation)
        plt.show()
    else:
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using deep reinforcement learning to train an agent on chrome dino")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--headless', action="store_true")

    args = parser.parse_args()

    main(args)




