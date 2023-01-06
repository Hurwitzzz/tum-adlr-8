import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from skimage.draw import line_aa
from stable_baselines3.a2c import A2C
from torch.utils.data import DataLoader

from recons_model.train import dir_mask, train_dir_img
from recons_model.unet import UNet
from recons_model.utils.data_loading import Tactile2dDataset


class Tactile2DEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, recons_model, dataloader, loss_fn):
        super(Tactile2DEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # (x,y) and x_pos, y_pos
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -0.5, -0.5]), high=np.array([0.5, 0.5, 0.5, 0.5]), dtype=np.float32
        )
        self.dataloader = dataloader
        self.image = None
        self.expected = None
        self.iter = 0

        self.loss_fn = loss_fn

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8)
        self.model = recons_model

    def step(self, action):

        x_dir, y_dir, x_pos, y_pos = action
        x_pos += 0.5
        y_pos += 0.5
        x_dir += 0.5
        y_dir += 0.5

        if abs(x_pos - 0.5) > abs(y_pos - 0.5):
            x_pos = round(x_pos)
        else:
            y_pos = round(y_pos)

        x_pos = int(x_pos * 99)
        y_pos = int(y_pos * 99)

        print(x_pos, y_pos)

        next_x, next_y = self._ray_cast(x_dir, y_dir, x_pos, y_pos)
        self.image[0, next_x, next_y] = 1

        recons = self.model(self.image)
        reward = -self.loss_fn(recons, self.expected)

        done = self.iter > 100

        return np.array(self.image), reward, done, {}

    def _ray_cast(self, x_dir, y_dir, x_pos, y_pos):
        """Using given starting position on the frame, and also the vector direction given
        casts the ray and finds the first intersecting position within the expected image

        Args:
            x_dir (int): x start
            y_dir (int): y start
            x_pos (float): x vector dir
            y_pos (float): y vector dir

        Returns:
            (np.ndarray): Position of the first intersection point
        """
        # starting point should be on the frame
        assert 0 in [x_pos, y_pos] or 99 in [x_pos, y_pos]
        self.iter += 1
        img = self.expected[0]

        if not x_dir and not y_dir:
            x_dir = 1
            y_dir = 1

        norm = np.linalg.norm(np.array([x_dir, y_dir]), ord=2)
        x_dir /= norm
        y_dir /= norm

        x_pos_end = x_pos
        y_pos_end = y_pos
        if 0 in [x_pos, y_pos]:
            while x_pos_end < 99 and y_pos_end < 99:
                x_pos_end += x_dir
                y_pos_end += y_dir

            x_pos_end, y_pos_end = int(x_pos_end), int(y_pos_end)
            rr, cc, _ = line_aa(x_pos, y_pos, x_pos_end, y_pos_end)
        else:
            while x_pos_end > 0 and y_pos_end > 0:
                x_pos_end -= x_dir
                y_pos_end -= y_dir

            x_pos_end, y_pos_end = int(x_pos_end), int(y_pos_end)
            rr, cc, _ = line_aa(x_pos_end, y_pos_end, x_pos, y_pos)

        masked = np.zeros_like(img)
        masked[rr, cc] = 1

        masked = np.logical_and(img, masked)
        indices = np.argwhere(masked)
        n_min = np.argmin(np.linalg.norm(indices - np.array([[x_pos], [y_pos]]), ord=2, axis=0))

        return indices[:, n_min]

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
    model = UNet(n_channels=1, n_classes=2, bilinear=False)
    # model.load_state_dict(torch.load(open("ckpt")))
    
    batch_size = 4
    train_set = Tactile2dDataset(train_dir_img, dir_mask, 1.0)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    for data in train_loader:
        plt.imshow(np.array(data))
        plt.imsave("1.png")
        break

    exit()

    dataloader = iter([torch.randint(0, 255, (1, 100, 100)), torch.randint(0, 255, (1, 100, 100))])
    loss_fn = lambda x, y: torch.randn((1,))

    env = Tactile2DEnv(model, dataloader, loss_fn)
    # Define and Train the agent
    model = A2C("CnnPolicy", env).learn(total_timesteps=1000)
