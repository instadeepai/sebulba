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

from typing import Any, Callable, NamedTuple, Tuple

import chex
import jax
import typing_extensions

Item: typing_extensions.TypeAlias = jax.Array
Params: typing_extensions.TypeAlias = chex.ArrayTree
Observation: typing_extensions.TypeAlias = chex.ArrayTree
ModelParams: typing_extensions.TypeAlias = Any
OptState: typing_extensions.TypeAlias = chex.ArrayTree
Action: typing_extensions.TypeAlias = jax.Array
Extra: typing_extensions.TypeAlias = chex.ArrayTree

Env = Any
EnvBuilder = Callable[[int], Env]


class Trajectory(NamedTuple):
    obs: Item
    dones: Observation
    actions: Action
    extras: Extra
    rewards: Item
    next_obs: Item


@chex.dataclass
class AgentState:
    params: Params


ActorFn = Callable[
    [ModelParams, Observation, chex.Array],
    Tuple[Action, Extra],
]

LearnFn = Callable[
    [AgentState, Trajectory, jax.random.KeyArray],
    Tuple[AgentState, Any],
]
