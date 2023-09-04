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

import math
import queue
from typing import Callable, List, Union

import jax

from sebulba import core, logging, types

ParamsChangeHandler = Callable[[types.Params], None]


class Learner(core.StoppableComponent):
    """
    `Learner` component, that retrieves trajectories from the `Pipeline` that are then used to
    carry out a learning update and updating the parameters of the `Actor`s.
    """

    def __init__(
        self,
        pipeline: core.Pipeline,
        local_devices: List[jax.Device],
        global_devices: List[jax.Device],
        init_state: types.AgentState,
        step_fn: types.LearnFn,
        key: jax.random.PRNGKeyArray,
        metrics_logger: logging.Hub,
        on_params_change: Union[List[Callable], None] = None,
    ):
        """Creates a `Learner` component that will shard its state across the given devices. The
        given step_fn is wrapped in a `pmap` to allow for batched learning across the devices.

        Args:
            pipeline: A pipeline to get trajectories from
            local_devices: local devices to use for learner
            global_devices: global devices that are part of the learning
            init_state: the initial state of the algorithm
            step_fn: the function to pmap that define the learning
            key: A PRNGKey for the jax computations
            metrics_logger: a logger to log to
            on_params_change: a list of callable to call when there is new params
                (this is typically used to update Actors params)
        Returns:
            A Learner that you can `start`, `stop` and `join`.
        """
        super().__init__(name="Learner")
        self.pipeline = pipeline
        self.local_devices = local_devices
        self.global_devices = global_devices
        self.state = jax.device_put_replicated(init_state, self.local_devices)
        self.step_fn_pmaped = jax.pmap(
            step_fn,
            "batch",
            devices=global_devices,
            in_axes=(0, 0, None),  # type: ignore
        )
        self.on_params_change = on_params_change
        self.rng = key
        self.metrics_logger = metrics_logger

    def _run(self) -> None:
        step = 0

        while not self.should_stop:
            try:
                batch = self.pipeline.get(block=True, timeout=1)
            except queue.Empty:
                continue
            else:
                with logging.RecordTimeTo(self.metrics_logger["step_time"]):
                    self.rng, key = jax.random.split(self.rng)
                    self.state, metrics = self.step_fn_pmaped(self.state, batch, key)

                    jax.tree_util.tree_map_with_path(
                        lambda path, value: self.metrics_logger[
                            f"agents/{'/'.join([p.key for p in path])}"
                        ].append(value[0].item()),
                        metrics,
                    )

                if self.on_params_change is not None:
                    new_params = jax.tree_map(lambda x: x[0], self.state.params)
                    for handler in self.on_params_change:
                        handler(new_params)

                step += 1

                self.metrics_logger["iteration"].add(1)
                self.metrics_logger["steps"].add(math.prod(batch.actions.shape))
                self.metrics_logger["queue_size"].append(self.pipeline.qsize())
