from typing import Callable, Optional, Union, Dict, List, Type, Tuple
import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        if "num_blocks" in kwargs.keys():
            self.num_blocks = kwargs["num_blocks"]
            del kwargs["num_blocks"]

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,  # type: ignore
            activation_fn,
            *args,
            **kwargs,
        )  # type: ignore

        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        if not self.num_blocks:
            raise AttributeError("num layers not specified")
        self.mlp_extractor = ScalableNetwork(feature_dim=self.observation_space.shape[0], num_blocks=self.num_blocks)  # type: ignore


class ScalableNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_blocks: int = 2,
    ):
        super(ScalableNetwork, self).__init__()

        self.num_blocks = num_blocks
        self.block_size = 64

        self.latent_dim_pi = self.block_size
        self.latent_dim_vf = self.block_size

        # Policy network
        self.pn_block = nn.Sequential(
            nn.Linear(self.block_size, self.block_size), nn.ReLU() #nn.BatchNorm1d(self.block_size), nn.ReLU()
        )
        pn_blocks = [self.pn_block for _ in range(num_blocks - 1)]  # final block with tanh is appended later

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, self.block_size),
            #nn.BatchNorm1d(self.block_size),
            nn.ReLU(),
            *pn_blocks,
            nn.Linear(self.block_size, self.block_size),
            #nn.BatchNorm1d(self.block_size),
            nn.Tanh(),
        )

        # Value network
        self.vn_block = nn.Sequential(
            nn.Linear(self.block_size, self.block_size),
            #nn.BatchNorm1d(self.block_size),
            nn.ReLU(),
        )
        vn_blocks = [self.vn_block for _ in range(num_blocks)]
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, self.block_size),
            #nn.BatchNorm1d(self.block_size),
            nn.ReLU(),
            *vn_blocks,
            nn.Linear(self.block_size, self.block_size),
            #nn.BatchNorm1d(self.block_size),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


env_id = "LunarLander-v2"  # "BipedalWalker-v3"
n_training_envs = 1
n_eval_envs = 10

eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

train_env_ppo = make_vec_env(env_id, n_envs=n_training_envs, seed=0)

eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0)

num_training_steps = 100_000
eval_frequency = 10_000


def train_model(model, eval_env, training_steps):
    results = {"rewards": [], "episode_lengths": [], "steps": []}

    for i in range(0, training_steps, eval_frequency):
        model.learn(eval_frequency)

        rewards, episode_lengths = evaluate_policy(model, eval_env, return_episode_rewards=True)
        results["rewards"].append(rewards)
        results["episode_lengths"].append(episode_lengths)
        results["steps"].append([i for _ in range(len(rewards))])

    results = {
        "rewards": np.array(results["rewards"]).flatten(),
        "episode_lengths": np.array(results["episode_lengths"]).flatten(),
        "steps": np.array(results["steps"]).flatten(),
    }

    return results

if __name__ == "__main__":
    network_sizes = [1, 2, 4, 8]

    result_df = pd.DataFrame(columns=["rewards", "steps", "episode_lengths", "network_size"])

    for n_size in network_sizes:
        model = PPO(CustomActorCriticPolicy, train_env_ppo, policy_kwargs={"num_blocks": n_size})

        res = train_model(model, eval_env, training_steps=num_training_steps)
        experiment_results = pd.DataFrame(res)
        experiment_results["network_size"] = n_size

        result_df = pd.concat([result_df, experiment_results])

    plt.figure(figsize=(15, 15))
    sns.lineplot(data=result_df, x="steps", y="rewards", hue="network_size")
    print(result_df.head(100))
    plt.savefig("experiment.jpg", dpi=300)
    plt.show()