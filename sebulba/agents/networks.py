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

"""
Flax implementation of NN from IMPALA without LSTM
sourced from cleanba:
https://github.com/vwxyzjn/cleanba/blob/main/cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py # noqa
"""  # noqa: E501

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import initializers


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Base(nn.Module):
    channels: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channels:
            x = ConvSequence(channels)(x)

        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            256,
            kernel_init=initializers.orthogonal(np.sqrt(2)),
            bias_init=initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(
            1,
            kernel_init=initializers.orthogonal(1),
            bias_init=initializers.constant(0.0),
        )(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(
            self.action_dim,
            kernel_init=initializers.orthogonal(0.01),
            bias_init=initializers.constant(0.0),
        )(x)
