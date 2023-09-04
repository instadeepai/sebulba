# Copyright 2023 Instadeep Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import List

import envpool
import omegaconf


def compute_num_envs(batch_size: int, learn_device: List) -> int:
    return len(learn_device) * batch_size


def get_env_specs(name: str) -> omegaconf.DictConfig:
    env = envpool.make(name, "gym")
    obs_shape = env.observation_space.shape
    n_action = env.action_space.n
    env.close()
    return omegaconf.OmegaConf.create(
        {
            "obs_shape": obs_shape,
            "n_action": n_action,
        }
    )


def register_resolver() -> None:
    """Register custom resolver to be used in config files"""

    omegaconf.OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)
    omegaconf.OmegaConf.register_new_resolver("len", len)
    omegaconf.OmegaConf.register_new_resolver("get_env_specs", get_env_specs)


@dataclasses.dataclass
class EnvConfig:
    name: str
    seed: int


@dataclasses.dataclass
class ActorConfig:
    enabled: bool
    devices: List[int]
    actor_per_device: int
    envs_per_actor: int
    traj_len: int


@dataclasses.dataclass
class LearnerConfig:
    enabled: bool
    devices: List[int]
    local_batch_size: int
    traj_len: int


@dataclasses.dataclass
class CommonConfig:
    traj_len: int
    batch_size: int


@dataclasses.dataclass
class Config:
    seed: int
    actor: ActorConfig
    env: EnvConfig
    learner: LearnerConfig
    common: CommonConfig
    pipeline: omegaconf.DictConfig
    loggers: omegaconf.DictConfig
    agents: omegaconf.DictConfig
    stopper: omegaconf.DictConfig
