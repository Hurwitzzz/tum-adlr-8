import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3.a2c import A2C


class Tactile2DEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, recons_model, dataloader, loss_fn):
        super(Tactile2DEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # (x,y) and x_pos, y_pos
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
        self.dataloader = dataloader
        self.image = torch.zeros((1, 100, 100))
        self.iter = 0
        self.expected = next(dataloader)
        self.loss_fn = loss_fn

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8)

        self.model = recons_model

    def step(self, action):

        x_dir, y_dir, x_pos, y_pos = action

        x_pos = int((x_pos+1) * 100)
        y_pos = int((y_pos+1) * 100)

        next_x, next_y = self._ray_cast(x_dir, y_dir, x_pos, y_pos)
        self.image[0, next_x, next_y] = 1

        recons = self.model(self.image)
        reward = -self.loss_fn(recons, self.expected)

        done = self.iter > 100

        return np.array(self.image), reward, done, {}

    def _ray_cast(self, x_dir, y_dir, x_pos, y_pos):
        return 10, 10

    def reset(self):
        self.image = torch.zeros((1, 100, 100), dtype=torch.uint8)
        self.expected = next(self.dataloader)
        self.iter = 0

        return np.array(self.image)

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # Instantiate the env

    model = lambda x: torch.ones((1, 100, 100))
    dataloader = iter([torch.zeros((1, 100, 100)), torch.ones((1, 100, 100))])
    loss_fn = lambda x, y: 1

    env = Tactile2DEnv(model, dataloader, loss_fn)
    # Define and Train the agent
    model = A2C("CnnPolicy", env).learn(total_timesteps=1000)