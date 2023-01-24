import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from stable_baselines3.a2c import A2C
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from torch.utils.data import DataLoader

import wandb
from recons_model.train import dir_mask, train_dir_img
from recons_model.unet import UNet
from recons_model.utils.data_loading import Tactile2dDataset
from recons_model.utils.dice_score import multiclass_dice_coeff

# from recons_model.utils.dice_score import dice_loss


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
                    self.exp.log(
                        {"num_timesteps": self.num_timesteps, "mean_reward": mean_reward, "step": self.global_Step}
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


class Tactile2DEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, recons_model, dataloader, loss_fn, device, experiment):
        super(Tactile2DEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # (x,y) and x_pos, y_pos
        self.exp = experiment

        self.action_space = spaces.Box(
            low=np.array([-1, -np.pi], dtype=np.float32), high=np.array([1, np.pi], dtype=np.float32), dtype=np.float32
        )
        self.dataloader_ = dataloader
        self.dataloader = iter(dataloader)
        self.device = device
        # self.expected = next(self.dataloader)["mask"]
        self.image = None
        self.expected = None
        self.iter = 0
        self.global_iter = 0

        self.loss_fn = loss_fn

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 100, 100), dtype=np.uint8)
        self.model = recons_model

    def step(self, action):

        pos, dir = action
        pos = int((pos + 1) * 199.5)

        if pos < 100:
            x_pos = pos
            y_pos = 0
        elif 100 <= pos < 200:
            x_pos = 99
            y_pos = pos - 100
        elif 200 <= pos < 300:
            x_pos = pos - 200
            y_pos = 99
        elif 300 <= pos < 400:
            x_pos = 0
            y_pos = pos - 300

        x_dir = np.cos(dir)
        y_dir = np.sin(dir)

        # print(pos)
        # print(x_pos, y_pos, x_dir, y_dir)
        next_x, next_y = self._ray_cast(x_dir, y_dir, x_pos, y_pos)
        reward = 0
        if next_x != -1 and next_y != -1 and self.image[0, next_x, next_y] != 1:
            self.image[0, next_x, next_y] = 1
            reward = 1
        else:
            pass
            # self.reward = np.clip(self.reward - 1, a_min=0, a_max=np.inf)

        recons = self.model(self.image[None, None, 0, ...].to(device=self.device, dtype=torch.float32))

        argmax_recons = torch.argmax(recons, dim=1)
        self.image[1, ...] = argmax_recons.detach().cpu()[0].type(torch.uint8)

        mask_pred = torch.functional.F.one_hot(argmax_recons, self.model.n_classes).permute(0, 3, 1, 2).float()
        mask_true = (
            torch.functional.F.one_hot(self.expected.to(self.device), self.model.n_classes).permute(0, 3, 1, 2).float()
        )
        # compute the Dice score, ignoring background
        coef = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=True)
        reward += coef.detach().cpu().item()
        if (self.global_iter % 1000) == 0:
            self.exp.log(
                {
                    "obs": {
                        "predicted_image": wandb.Image(self.image[1]),
                        "expected_image": wandb.Image(self.expected[0]),
                        "sample_points": wandb.Image(self.image[0]),
                    },
                    "step": self.global_iter,
                }
            )

        # + dice_loss(
        #     torch.functional.F.softmax(recons, dim=1).float(),
        #     torch.functional.F.one_hot(self.expected, 2).permute(0, 3, 1, 2).float(),
        #     multiclass=True,
        # )
        # print(loss.item())

        self.iter += 1
        self.global_iter += 1
        done = self.iter > 14

        return (
            np.array(self.image, dtype=np.uint8) * 255,
            reward,
            done,
            {},
        )

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

        norm = np.linalg.norm(np.array([x_dir, y_dir]), ord=2)
        x_dir /= norm
        y_dir /= norm

        while (x_pos <= 99 and y_pos <= 99) and (x_pos >= 0 and y_pos >= 0):
            x_pos += x_dir
            y_pos += y_dir

            x, y = np.clip(int(x_pos), a_min=0, a_max=99, dtype=int), np.clip(int(y_pos), a_min=0, a_max=99, dtype=int)
            if img[x, y]:
                return x, y

        return -1, -1

    def reset(self):
        self.image = torch.zeros((2, 100, 100), dtype=torch.uint8)
        try:
            batch = next(self.dataloader)
        except Exception:
            self.dataloader = iter(self.dataloader_)
            batch = next(self.dataloader)
        self.expected = batch["mask"].to(dtype=torch.long)
        self.iter = 0

        return np.array(self.image, dtype=np.uint8)

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    lr = 1e-3
    n_steps = 14 * 128
    total_timestamps = 50000
    n_env = 4
    device = "cuda"
    check_freq = 1000

    experiment = wandb.init(project="tactile experiment", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            total_timestamps=total_timestamps, n_steps=n_steps, n_env=n_env, lr=lr, device=device, check_freq=check_freq
        )
    )

    model = UNet(n_channels=1, n_classes=2, bilinear=False)

    model.to(device)
    model.load_state_dict(torch.load("./checkpoints/INTERRUPTED.pth", map_location=torch.device(device)))

    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Logs will be saved in log_dir/monitor.csv
    batch_size = 1
    train_set = Tactile2dDataset(train_dir_img, dir_mask, 1.0)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    loss_fn = torch.nn.CrossEntropyLoss()

    env = make_vec_env(
        Tactile2DEnv,
        n_envs=4,
        monitor_dir=log_dir,
        env_kwargs={
            "recons_model": model,
            "dataloader": train_loader,
            "loss_fn": loss_fn,
            "device": device,
            "experiment": experiment,
        },
    )
    # env = Monitor(env, log_dir)

    # Define and Train the agent

    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, experiment=experiment)

    model = A2C(
        "CnnPolicy",
        env,
        n_steps=n_steps,
        use_rms_prop=False,
        learning_rate=lr,
        # batch_size=16,
        # n_epochs=10, learning_rate=3e-4,
        seed=0,
        device=device,
    )
    if os.path.exists("/tmp/gym/best_model.zip"):
        model.load("/tmp/gym/best_model.zip", env=env, device=device)
    model.learn(total_timesteps=total_timestamps, callback=callback)

    results_plotter.plot_results([log_dir], total_timestamps, results_plotter.X_TIMESTEPS, "Tactile")
    plot_results(log_dir)
