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

from typing import Sequence, Tuple

import jax
import numpy as np

from sebulba import config, core, logging, types


class Actor(core.StoppableComponent):
    """Actor Component that generates trajectories by passing a batch of actions
    to our batched environment, which then outputs the corresponding observations. These
    are then used to form trajectories that are placed in our pipeline to be feed to our
    `Learner`.
    """

    def __init__(
        self,
        env_builder: types.EnvBuilder,
        actor_device: jax.Device,
        params_source: core.ParamsSource,
        pipeline: core.Pipeline,
        actor_fn: types.ActorFn,
        key: jax.random.PRNGKeyArray,
        config: config.ActorConfig,
        metrics_logger: logging.Hub,
        name: str,
    ):
        """
        Create an `Actor` component that will initialise a batch of environments and pipeline to
        pass trajectories to a `Learner` component.

        Args:
            env_builder: An Callable that take an int and return a vector env
            actor_device: The jax device to use for the policy
            params_source: A source of parameter
            pipeline: The pipeline to put trajectories
            actor_fn: An ActorFn to generate action from observations
            key: A PRNGKey for the jax computations
            config: The Actor configuration
            metrics_logger: The base logger to use for this component, a sublogger with name `name`
                will be created
            name: The name of this actor, used for thread name and logging path
        Returns:
            An actor that you can `start`, `stop` and `join`
        """
        super().__init__(name=f"Actor-{name}")
        self.actor_device = actor_device
        self.pipeline = pipeline
        self.params_source = params_source
        self.actor_fn = actor_fn
        self.rng = jax.device_put(key, actor_device)
        self.config = config
        self.metrics_logger = metrics_logger.create_sub(self.name)
        self.cpu = jax.devices("cpu")[0]

        self.envs = env_builder(config.envs_per_actor)

    def _run(self) -> None:
        episode_return = np.zeros((self.config.envs_per_actor,))
        with jax.default_device(self.actor_device):

            @jax.jit
            def wrapped_act_fn(
                params: types.ModelParams,
                key: jax.random.KeyArray,
                next_obs: types.Observation,
            ) -> Tuple[Tuple[types.Action, types.Extra], jax.random.KeyArray]:
                key, next_key = jax.random.split(key)
                action, extra = self.actor_fn(params, next_obs, key)
                return (action, extra), (next_key, next_obs)

            obs, _ = self.envs.reset()
            obs = jax.device_put(obs, self.actor_device)

            while not self.should_stop:
                traj_obs = []
                traj_dones = []
                traj_actions = []
                traj_extras = []
                traj_rewards = []

                for _t in range(self.config.traj_len):
                    params = self.params_source.get()
                    with logging.RecordTimeTo(self.metrics_logger["wrapped_act_fn"]):
                        (action, extra), (self.rng, obs) = wrapped_act_fn(
                            params, self.rng, obs
                        )

                    with logging.RecordTimeTo(self.metrics_logger["get_cpu"]):
                        action_cpu = np.array(jax.device_put(action, self.cpu))

                    with logging.RecordTimeTo(self.metrics_logger["env/step_time"]):
                        next_obs, reward, terminated, truncated, info = self.envs.step(
                            action_cpu
                        )

                    dones = terminated | truncated

                    # Here we use the reward in info because this one is not clipped
                    episode_return += info["reward"]

                    traj_obs.append(obs)
                    traj_dones.append(dones)
                    traj_actions.append(action)
                    traj_extras.append(extra)
                    traj_rewards.append(reward)

                    for env_idx, env_done in enumerate(dones):
                        if env_done:
                            self.metrics_logger["episode_return"].append(
                                episode_return[env_idx]
                            )
                            self.metrics_logger["episode_len"].append(
                                info["elapsed_step"][env_idx]
                            )
                    self.metrics_logger["episode"].add(np.sum(dones))
                    # Here we use terminated and not done to handle episodic_life
                    episode_return *= 1.0 - info["terminated"]

                    obs = next_obs

                self.metrics_logger["steps"].add(
                    self.config.envs_per_actor * self.config.traj_len
                )

                self.process_item(
                    traj_obs, traj_dones, traj_actions, traj_extras, traj_rewards, obs
                )

    def process_item(
        self,
        traj_obs: Sequence[types.Observation],
        traj_dones: Sequence[jax.Array],
        traj_actions: Sequence[jax.Array],
        traj_extras: Sequence[jax.Array],
        traj_rewards: Sequence[jax.Array],
        next_obs: types.Observation,
    ) -> None:
        """Process a trajectory and put it in the pipeline."""
        with logging.RecordTimeTo(self.metrics_logger["pipeline_put_time"]):
            self.pipeline.put(
                traj_obs, traj_dones, traj_actions, traj_extras, traj_rewards, next_obs
            )
