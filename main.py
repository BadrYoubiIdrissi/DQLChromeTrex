import argparse
import gym
import chrome_trex_gym
from DeepDino.train import train
from DeepDino.test import test

def main(args):
    try:
        env = gym.make("chrome-trex-v0")
        if not args.test:
            train(env)
        else:
            test(args.test, env)
    finally:
        env.game.close()
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using deep reinforcement learning to train an agent on chrome dino")
    parser.add_argument('--test')

    args = parser.parse_args()

    main(args)




