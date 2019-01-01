from ..DinoGame import DinoGame
import gym
from gym.spaces import Discrete, Box
import numpy as np
from skimage.transform import resize
class ChromeTrexEnv(gym.Env):
    def __init__(self, FPS=60, headless=False):
        self.action_space = Discrete(3)
        scr_size = (150,600)
        self.observation_space = Box(low=0, high=255, shape=scr_size, dtype=np.uint8)
        self.game = DinoGame(FPS=FPS, headless=headless)
        self.game.step(0)
        self.observation = [self.getTransformedImage() for i in range(4)]

    def getTransformedImage(self):
        image = self.game.get_image()[:,:,0].T
        return resize(image[50:,:], (100, 100),anti_aliasing=False, mode="constant")

    def step(self, action):
        if not self.game.gameOver:
            self.game.step(action)
        reward = -1 if self.game.gameOver else 1
        self.observation.pop(0)
        self.observation.append(self.getTransformedImage())
        return self.getStackedObservation(), reward, self.game.gameOver, {}

    def getStackedObservation(self):
        return np.stack(self.observation, axis=-1)

    def reset(self):
        self.game.reset()
        self.game.step(0)
        self.observation = [self.getTransformedImage() for i in range(4)]

    def __enters__(self):
        return self
    
    def __exits__(self, type, value, traceback):
        self.game.close()