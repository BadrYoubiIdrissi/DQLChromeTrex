import argparse
import gym
import chrome_trex_gym
from DeepDino.train import train

def main(args):
    if not args.test:
        try:
            env = gym.make("chrome-trex-v0")
            train(env)
        finally:
            env.game.close()
    else:
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using deep reinforcement learning to train an agent on chrome dino")
    parser.add_argument('--test', action="store_true")

    args = parser.parse_args()

    main(args)




