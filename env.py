import os
import time
from typing import Callable, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from PIL import Image
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.ppo import PPO
from torch.utils.data import DataLoader

import wandb
from recons_model.train import dir_mask, train_dir_img
from recons_model.unet import UNet
from recons_model.utils.data_loading import Tactile2dDataset
from recons_model.utils.dice_score import multiclass_dice_coeff

# from recons_model.utils.dice_score import dice_loss


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    _ = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig("plot.png")


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        experiment,
        verbose=1,
    ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.exp = experiment

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} -"
                        + f" Last mean reward per episode: {mean_reward:.2f}"
                    )
                    self.exp.log({"num_timesteps": self.num_timesteps, "mean_reward": mean_reward})

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path + str(experiment.name))

        return True


class Tactile2DEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, recons_model, dataset, device, experiment, loader_args):
        super(Tactile2DEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # (x,y) and x_pos, y_pos
        self.exp = experiment

        """ self.action_space = spaces.Box(
            low=np.array([-1, -1, -np.pi / 2], dtype=np.float32),
            high=np.array([1, 1, np.pi / 2], dtype=np.float32),
            dtype=np.float32,
        ) """

        self.action_space = spaces.MultiDiscrete((4, 100, 180))

        self.dataloader_ = DataLoader(dataset, shuffle=True, **loader_args)
        self.dataloader = iter(self.dataloader_)
        self.device = device
        # self.expected = next(self.dataloader)["mask"]
        self.image = None
        self.expected = None
        self.iter = 0
        self.global_iter = 0
        self.coef = 0
        self.ray_images = []

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict(
            {
                "sample_points": spaces.Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8),
                "reconstruction": spaces.Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8),
                "ray": spaces.Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8),
            }
        )
        self.model = recons_model

    def convert_discrete_actions_into_meaningful_actions(self, action):
        side, offset, dir = action
        """ dir += np.pi / 2
        side = int(np.clip(np.rint((side + 0.75) * 2), a_min=0, a_max=3))  # -0.25 - 1.75 -> -0.5 - 3.5
        offset = int(np.clip(np.rint((offset + 0.99) * 50), a_min=0, a_max=99)) """
        pos = side * 100 + offset
        dir = np.deg2rad(dir)

        if pos < 100:  # top
            x_pos = pos
            y_pos = 0
            dir = dir
        elif 100 <= pos < 200:  # right
            x_pos = 99
            y_pos = pos - 100
            dir += np.pi / 2
        elif 200 <= pos < 300:  # bottom
            x_pos = pos - 200
            y_pos = 99
            dir += np.pi
        elif 300 <= pos < 400:  # left
            x_pos = 0
            y_pos = pos - 300
            dir -= np.pi / 2

        x_dir = np.cos(dir)
        y_dir = np.sin(dir)

        # self.prev_pos[self.iter, :] = np.array([x_pos, y_pos])
        # self.prev_dir[self.iter, :] = np.array([x_dir, y_dir])

        return x_dir, y_dir, x_pos, y_pos

    def step(self, action):
        x_dir, y_dir, x_pos, y_pos = self.convert_discrete_actions_into_meaningful_actions(action)

        # define reward
        reward = 0

        # mark starting position
        self.image[2, x_pos, y_pos] = 1

        # get sampled point and mark ray within function
        next_x, next_y = self._ray_cast(x_dir, y_dir, x_pos, y_pos)

        # if point is sampled and new
        if next_x != -1 and next_y != -1 and self.image[0, next_x, next_y] != 1:
            # mark hit
            self.image[0, next_x, next_y] = 1
            reward += self.get_reward()

        self.log()

        self.iter += 1
        self.global_iter += 1

        done = self.iter == 14

        return (
            {
                "sample_points": np.array(self.image[0:1] * 255, dtype=np.uint8),
                "reconstruction": np.array(self.image[1:2] * 255, dtype=np.uint8),
                "ray": np.array(self.image[2:] * 255, dtype=np.uint8),
            },
            reward,
            done,
            {},
        )

    def log(self, n_iter=1000):
        if self.global_iter % n_iter < 27 and self.iter < 14:
            self.ray_images.append(Image.fromarray(self.image[2].float()))
            if self.iter == 13:
                name = f"./tmp/gifs/{time.time()}.gif"
                self.ray_images[0].save(
                    name, save_all=True, append_images=self.ray_images[1:], optimize=False, duration=14, loop=1
                )

                self.exp.log(
                    {
                        "obs": {
                            "predicted_image_foreground": wandb.Image(self.image[1].float()),
                            "expected_image": wandb.Image(self.expected[0].float()),
                            "sample_points": wandb.Image(self.image[0].float()),
                            "ray_points": wandb.Image(self.image[2].float()),
                            "ray": wandb.Video(name, fps=1, format="gif"),
                        },
                        "step": self.global_iter,
                        "dice_coef": self.coef,
                    }
                )

    def get_reward(self):
        # generate reconstruction and calculate reward
        recons = self.model(self.image[None, None, 0, ...].to(device=self.device, dtype=torch.float32))

        softmax_recons = torch.nn.functional.softmax(recons, dim=1)
        self.image[1, ...] = softmax_recons[0].detach().cpu()

        mask_true = (
            torch.functional.F.one_hot(self.expected.to(self.device), self.model.n_classes).permute(0, 3, 1, 2).float()
        )

        coef = multiclass_dice_coeff(softmax_recons[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=True)
        coef = coef.detach().cpu().item()
        self.coef = coef
        return coef

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

        img = self.expected[0]

        while (x_pos <= 99 and y_pos <= 99) and (x_pos >= 0 and y_pos >= 0):
            x_pos += x_dir
            y_pos += y_dir

            x, y = np.clip(int(x_pos), a_min=0, a_max=99, dtype=int), np.clip(int(y_pos), a_min=0, a_max=99, dtype=int)
            # plot the ray for observation
            self.image[2, x, y] = 1
            if img[x, y]:
                return x, y

        return -1, -1

    def reset(self):
        self.image = torch.zeros((3, 100, 100), dtype=torch.float32)
        try:
            batch = next(self.dataloader)
        except Exception:
            self.dataloader = iter(self.dataloader_)
            batch = next(self.dataloader)
        self.expected = batch["mask"].to(dtype=torch.long)
        self.iter = 0
        self.coef = 0
        self.ray_images = []

        return {
            "sample_points": np.array(self.image[0:1] * 255, dtype=np.uint8),
            "reconstruction": np.array(self.image[1:2] * 255, dtype=np.uint8),
            "ray": np.array(self.image[2:] * 255, dtype=np.uint8),
        }

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    lr = 7e-4
    n_steps = 5
    total_timestamps = 5000000
    n_env = 2
    device = "cuda"
    check_freq = 1000

    experiment = wandb.init(project="tactile experiment", name="MultiInputDiscrete", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            total_timestamps=total_timestamps, n_steps=n_steps, n_env=n_env, lr=lr, device=device, check_freq=check_freq
        )
    )

    model = UNet(n_channels=1, n_classes=2, bilinear=False)

    model.to(device)
    model.load_state_dict(torch.load("./checkpoints/INTERRUPTED.pth", map_location=torch.device(device)))
    model.eval()

    # Create log dir
    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Logs will be saved in log_dir/monitor.csv
    batch_size = 1
    train_set = Tactile2dDataset(train_dir_img, dir_mask, 1.0)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    env = make_vec_env(
        Tactile2DEnv,
        n_envs=n_env,
        monitor_dir=log_dir,
        env_kwargs={
            "recons_model": model,
            "dataset": train_set,
            "device": device,
            "experiment": experiment,
            "loader_args": loader_args,
        },
    )
    # env = Monitor(env, log_dir)

    # Define and Train the agent

    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, experiment=experiment)
    model = PPO(
        "MultiInputPolicy",
        env,
        # n_steps=n_steps,
        # use_rms_prop=False,
        learning_rate=linear_schedule(lr),
        # batch_size=16,
        # n_epochs=10, learning_rate=3e-4,
        seed=0,
        device=device,
    )
    print(experiment.name)
    model.learn(total_timesteps=total_timestamps, callback=callback)

    results_plotter.plot_results([log_dir], total_timestamps, results_plotter.X_TIMESTEPS, "Tactile")
    plot_results(log_dir)
