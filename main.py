import argparse
import gym
import chrome_trex_gym
from DeepDino.train import train
from DeepDino.test import test

def main(args):
    try:
        if args.test:
            env = gym.make("chrome-trex-v0")
            test(args.test, env)
        else:
            if args.headless:
                env = gym.make("chrome-trex-train-v0")
            else:
                env = gym.make("chrome-trex-train-render-v0")
            train(env)

    finally:
        env.game.close()
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using deep reinforcement learning to train an agent on chrome dino")
    parser.add_argument('--test')
    parser.add_argument('--headless')

    args = parser.parse_args()

    main(args)




