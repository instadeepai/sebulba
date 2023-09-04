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

import logging
import threading
from typing import Any

import envpool
import hydra
import jax
import omegaconf

from sebulba import actor as sebulba_actor
from sebulba import config, core
from sebulba import learner as sebulba_learner
from sebulba import logging as sebulba_logging
from sebulba import stoppers, types, utils

ATARI_MAX_FRAMES = int(108000 / 4)


class EnvPoolFactory:
    """
    Create environments with different seeds for each `Actor`
    """

    def __init__(self, init_seed: int = 42, **kwargs: Any):
        self.seed = init_seed
        # a lock is needed because this object will be used from different threads.
        # We want to make sure all seeds are unique
        self.lock = threading.Lock()
        self.kwargs = kwargs

    def __call__(self, num_envs: int) -> types.Env:
        with self.lock:
            seed = self.seed
            self.seed += num_envs
            return envpool.make(**self.kwargs, num_envs=num_envs, seed=seed)


@hydra.main(config_path="config", version_base=None, config_name="base")
def run(cfg: config.Config):  # noqa CCR001
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.info(f"Script: {__file__}")
    log.info(f"Imported hydra config:\n{omegaconf.OmegaConf.to_yaml(cfg)}")
    log.info(f"[JAX] Local devices: {jax.local_devices()}.")
    log.info(f"[JAX] Global devices: {jax.devices()}.")

    rng_key = jax.random.PRNGKey(cfg.seed)
    learner_state_key, learner_key, actors_key = jax.random.split(rng_key, 3)

    # Get jax devices local and global for learning
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    actor_devices = [local_devices[device_id] for device_id in cfg.actor.devices]
    local_learner_devices = [
        local_devices[device_id] for device_id in cfg.learner.devices
    ]
    global_learner_devices = [
        device
        for i, device in enumerate(global_devices)
        if i % len(local_devices) in cfg.learner.devices
    ]

    log.info(f"[Sebulba] Actors devices: {actor_devices}")
    log.info(f"[Sebulba] Learner devices: {local_learner_devices}")

    # Create environments factory, uses the same setup as seen in CleanRL
    env_factory = EnvPoolFactory(
        cfg.env.seed,
        task_id=cfg.env.name,
        env_type="gym",
        episodic_life=True,
        repeat_action_probability=0,
        noop_max=30,
        full_action_space=False,
        max_episode_steps=ATARI_MAX_FRAMES,
        reward_clip=True,
    )

    # build Pipeline to pass trajectories between Actor's and Learner
    partial_pipeline = hydra.utils.instantiate(cfg.pipeline)
    pipeline: core.Pipeline = partial_pipeline(learner_devices=local_learner_devices)
    pipeline.start()

    # Setup logging
    # we only log on main process
    if jax.process_index() == 0:
        recorder: sebulba_logging.Recorder = hydra.utils.instantiate(cfg.loggers)
        recorder.record_info("config", cfg)
        info = utils.get_info()

        for k, v in info.items():
            recorder.record_info(f"info/{k}", v)
    else:
        recorder = sebulba_logging.Recorder()

    logger_manager = sebulba_logging.LoggerManager(recorder)
    logger_manager.start()

    # Build out agent from config
    agent_builder = hydra.utils.instantiate(cfg.agents)

    learn_fn = agent_builder.build_learn_fn()
    learn_state = agent_builder.build_state(learner_state_key)

    actors = []
    params_sources = []
    actors_logger = logger_manager["actors"]
    actor_fn = agent_builder.build_actor_fn()

    for actor_device in actor_devices:
        # Create 1 params source per actor device
        params_source = core.ParamsSource(learn_state.params, actor_device)
        params_source.start()
        params_sources.append(params_source)

        for i in range(cfg.actor.actor_per_device):
            actors_key, key = jax.random.split(actors_key)
            # Create Actors
            actor = sebulba_actor.Actor(
                env_factory,
                actor_device,
                params_source,
                pipeline,
                actor_fn,
                key,
                cfg.actor,
                actors_logger,
                f"{actor_device.id}-{i}",
            )
            actors.append(actor)

    # Create Learner
    learner = sebulba_learner.Learner(
        pipeline,
        local_learner_devices,
        global_learner_devices,
        learn_state,
        learn_fn,
        learner_key,
        logger_manager["learner"],
        on_params_change=[params_source.update for params_source in params_sources],
    )

    # Start Learner and Actors
    learner.start()
    for actor in actors:
        actor.start()
    try:
        # Create our stopper and wait for it to stop
        stopper: stoppers.Stopper = hydra.utils.instantiate(cfg.stopper)(
            logger_manager=logger_manager
        )
        stopper.wait()
    finally:
        log.info("Shutting down")
        # Try to gracefully shutdown all components
        # If not finished after 10 second exits anyway

        def graceful_shutdown():
            for actor in actors:
                actor.stop()
            for params_source in params_sources:
                params_source.stop()

            for actor in actors:
                actor.join()

            learner.stop()

            learner.join()
            pipeline.stop()
            pipeline.join()
            logger_manager.stop()
            for params_source in params_sources:
                params_source.join()
            logger_manager.join()

        graceful_thread = threading.Thread(target=graceful_shutdown, daemon=True)
        graceful_thread.start()
        graceful_thread.join(timeout=10.0)
        if graceful_thread.is_alive():
            log.warning("Shutdown was not graceful")

    log.info("Training finished!")


if __name__ == "__main__":
    config.register_resolver()
    run()
