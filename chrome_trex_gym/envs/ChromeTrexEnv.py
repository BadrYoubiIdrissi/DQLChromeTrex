from ..DinoGame import DinoGame
import gym
from gym.spaces import Discrete, Box
import numpy as np
class ChromeTrexEnv(gym.Env):
    def __init__(self):
        self.game = DinoGame(FPS=0)
        self.action_space = Discrete(3)
        scr_size = (150,600)
        self.observation_space = Box(low=0, high=255, shape=scr_size)
        self.observation = [np.zeros(scr_size) for i in range(4)]

    def step(self, action):
        if not self.game.gameOver:
            self.game.step(action)
        reward = -1 if self.game.gameOver else 1
        self.observation.pop(0)
        self.observation.append(self.game.get_image()[:,:,0].T)
        return self.observation, reward, self.game.gameOver, {}

    def reset(self):
        self.game.reset()

    def __enters__(self):
        return self
    
    def __exits__(self, type, value, traceback):
        self.game.close()
