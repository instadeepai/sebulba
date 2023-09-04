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
PPO implementation adapted from:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
"""

import dataclasses
import functools
from typing import Any, List, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from sebulba import types
from sebulba.agents import networks


@dataclasses.dataclass
class PPOConfig:
    n_action: int
    obs_shape: List[int]
    max_grad_norm: float
    anneal_lr: bool
    learning_rate: float
    num_minibatches: int
    update_epochs: int
    total_timesteps: int
    norm_adv: bool
    clip_coef: float
    ent_coef: float
    vf_coef: float
    gae_lambda: float
    traj_len: int
    batch_size: int
    num_learner_device: int
    gamma: float


@chex.dataclass
class ActorCriticModelParams:
    base_params: Any
    actor_params: Any
    critic_params: Any


class PPOAgentBuilder:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.base_network = networks.Base()
        self.actor_network = networks.Actor(self.config.n_action)
        self.critic_network = networks.Critic()

    def _build_optimizer(self) -> optax.GradientTransformation:
        def linear_schedule(count: int) -> float:
            # anneal learning rate linearly after one training iteration which contains
            # (args.num_minibatches * args.update_epochs) gradient updates
            frac = 1.0 - (
                count // (self.config.num_minibatches * self.config.update_epochs)
            ) / (
                self.config.total_timesteps
                / (
                    self.config.batch_size
                    * self.config.traj_len
                    * self.config.num_learner_device
                )
            )
            return self.config.learning_rate * frac  # type: ignore

        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule
                if self.config.anneal_lr
                else self.config.learning_rate,
                eps=1e-5,
            ),
        )

    def build_state(self, key: jax.random.KeyArray) -> types.AgentState:
        key_network, key_actor, key_critic = jax.random.split(key, 3)
        sample_input = jnp.zeros((1, *self.config.obs_shape))
        base_params = self.base_network.init(key_network, sample_input)
        network_output = self.base_network.apply(base_params, sample_input)

        return train_state.TrainState.create(  # type: ignore
            apply_fn=None,
            params=ActorCriticModelParams(  # type: ignore
                base_params=base_params,
                actor_params=self.actor_network.init(key_actor, network_output),
                critic_params=self.critic_network.init(key_critic, network_output),
            ),
            tx=self._build_optimizer(),
        )

    def build_actor_fn(self) -> types.ActorFn:
        def get_action_and_value(
            params: ActorCriticModelParams,
            next_obs: types.Observation,
            key: jax.random.KeyArray,
        ) -> Tuple[types.Action, types.Extra]:
            next_obs = jnp.array(next_obs)
            hidden = networks.Base().apply(params.base_params, next_obs)
            logits = networks.Actor(self.config.n_action).apply(
                params.actor_params, hidden
            )
            # sample action: Gumbel-softmax trick
            key, subkey = jax.random.split(key)
            sample = jax.random.uniform(subkey, shape=logits.shape)
            action = jnp.argmax(logits - jnp.log(-jnp.log(sample)), axis=1)
            logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
            value = networks.Critic().apply(params.critic_params, hidden)
            return action, {
                "logprobs": logprob,
                "values": value.squeeze(),
            }

        return get_action_and_value

    def build_learn_fn(self) -> types.LearnFn:
        return functools.partial(
            step_fn,
            self.critic_network,
            self.base_network,
            self.config,
        )


def get_action_and_value2(
    params: ActorCriticModelParams,
    x: types.Observation,
    action: types.Action,
    action_dim: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    hidden = networks.Base().apply(params.base_params, x)
    logits = networks.Actor(action_dim).apply(params.actor_params, hidden)

    all_logprob = jax.nn.log_softmax(logits)

    logprob = all_logprob[jnp.arange(action.shape[0]), action]
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    entropy = -p_log_p.sum(-1)
    value = networks.Critic().apply(params.critic_params, hidden).squeeze()
    return logprob, entropy, value


def ppo_loss(
    params: ActorCriticModelParams,
    observations: types.Observation,
    actions: types.Action,
    original_log_probs: jax.Array,
    mb_advantages: jax.Array,
    mb_returns: jax.Array,
    action_dim: int,
    clip_coef: float,
    norm_adv: bool,
    ent_coef: float,
    vf_coef: float,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    new_log_probs, entropy, new_value = get_action_and_value2(
        params, observations, actions, action_dim
    )
    log_ratio = new_log_probs - original_log_probs
    ratio = jnp.exp(log_ratio)

    clip_frac = jnp.mean(jnp.abs(1.0 - ratio) > clip_coef)

    approx_kl = ((ratio - 1) - log_ratio).mean()

    if norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
    return loss, (
        pg_loss,
        v_loss,
        entropy_loss,
        jax.lax.stop_gradient(approx_kl),
        clip_frac,
    )


@functools.partial(jax.jit, static_argnums=(7,))
def single_device_update(
    agent_state: types.AgentState,
    b_obs: types.Observation,
    b_actions: types.Action,
    b_logprobs: jax.Array,
    b_advantages: jax.Array,
    b_returns: jax.Array,
    key: jax.random.KeyArray,
    config: PPOConfig,
) -> Tuple[types.AgentState, Any, Any, Any, Any, Any, Any, jax.random.KeyArray]:
    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def update_epoch(
        carry: Tuple[types.AgentState, jax.random.KeyArray], _: int
    ) -> Tuple[
        Tuple[types.AgentState, jax.random.KeyArray],
        Tuple[Any, Any, Any, Any, Any, Any, Any],
    ]:
        agent_state, key = carry
        key, subkey = jax.random.split(key, 2)

        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jax.Array) -> jax.Array:
            x = jax.random.permutation(subkey, x)
            x = jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:])
            return x

        def update_minibatch(
            agent_state: types.AgentState,
            minibatch: Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> Tuple[types.AgentState, Tuple[Any, Any, Any, Any, Any, Any, Any]]:
            mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns = minibatch
            (
                loss,
                (pg_loss, v_loss, entropy_loss, approx_kl, clip_frac),
            ), grads = ppo_loss_grad_fn(
                agent_state.params,
                mb_obs,
                mb_actions,
                mb_logprobs,
                mb_advantages,
                mb_returns,
                config.n_action,
                config.clip_coef,
                config.norm_adv,
                config.ent_coef,
                config.vf_coef,
            )
            grads = jax.lax.pmean(grads, axis_name="batch")
            agent_state = agent_state.apply_gradients(grads=grads)  # type: ignore
            return agent_state, (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                clip_frac,
                grads,
            )

        agent_state, (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            clip_frac,
            grads,
        ) = jax.lax.scan(
            update_minibatch,
            agent_state,
            (
                convert_data(b_obs),
                convert_data(b_actions),
                convert_data(b_logprobs),
                convert_data(b_advantages),
                convert_data(b_returns),
            ),
        )
        return (agent_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            clip_frac,
            grads,
        )

    (agent_state, key), (
        loss,
        pg_loss,
        v_loss,
        entropy_loss,
        approx_kl,
        clip_frac,
        _,
    ) = jax.lax.scan(update_epoch, (agent_state, key), (), length=config.update_epochs)
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, clip_frac, key


def step_fn(
    critic: networks.Critic,
    network: networks.Base,
    cfg: PPOConfig,
    state: train_state.TrainState,
    batch: types.Trajectory,
    key: jax.random.KeyArray,
) -> Tuple[types.AgentState, Any]:
    # preprocess
    b_obs = batch.obs
    b_actions = batch.actions
    b_logprobs = batch.extras["logprobs"]
    b_dones = batch.dones
    b_rewards = batch.rewards
    b_values = batch.extras["values"]

    next_values = critic.apply(
        state.params.critic_params,
        network.apply(state.params.base_params, batch.next_obs),
    ).squeeze()
    next_values = jnp.expand_dims(next_values, 0)

    advantages = jnp.zeros((b_rewards.shape[1],))
    values = jnp.concatenate([b_values, next_values], axis=0)
    _, b_advantages = jax.lax.scan(
        functools.partial(
            compute_gae_once,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        ),
        advantages,
        (b_dones, values[1:], values[:-1], b_rewards),
        reverse=True,
    )
    b_returns = b_advantages + b_values

    def flatten(x: jax.Array) -> jax.Array:
        return x.reshape((-1,) + x.shape[2:])

    (
        state,
        loss,
        pg_loss,
        v_loss,
        entropy_loss,
        approx_kl,
        clip_frac,
        key,
    ) = single_device_update(
        state,
        flatten(b_obs),
        flatten(b_actions),
        flatten(b_logprobs),
        flatten(b_advantages),
        flatten(b_returns),
        key,
        cfg,
    )

    def mean_all(x: jax.Array) -> jax.Array:
        return jax.lax.pmean(jnp.mean(x), axis_name="batch")

    metrics = {
        "loss": loss,
        "pg_loss": pg_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "learning_rate": state.opt_state[1].hyperparams["learning_rate"],
    }
    return state, jax.tree_map(mean_all, metrics)


def compute_gae_once(
    carry: jax.Array,
    inp: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    gamma: float,
    gae_lambda: float,
) -> Tuple[jax.Array, jax.Array]:
    advantages = carry
    nextdone, nextvalues, curvalues, reward = inp
    nextnonterminal = 1.0 - nextdone

    delta = reward + gamma * nextvalues * nextnonterminal - curvalues
    advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
    # we return 2 time the value here because this function is used with jax.lax.scan
    # the first value is passed to the new call and the second for the result.
    return advantages, advantages
