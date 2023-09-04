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

import abc
import time
from typing import Any, Union

import omegaconf

from sebulba import logging


class Stopper(abc.ABC):
    """
    A Stopper is in charge of stopping the system during training.
    """

    @abc.abstractmethod
    def wait(self) -> None:
        pass


class TimeStopper(Stopper):
    """
    A `Stopper` that stops the system after a certain amount of time.
    """

    def __init__(self, config: omegaconf.DictConfig, **kargs: Any):
        self.wait_time = config.wait_time

    def wait(self) -> None:
        time.sleep(self.wait_time)


class LearnerStepStopper(Stopper):
    """
    A `Stopper` that stops the system after a certain number of learner steps.
    """

    def __init__(
        self, config: omegaconf.DictConfig, logger_manager: logging.LoggerManager
    ):
        self.num_steps = config.num_steps
        self.logger_manager = logger_manager

    def get_current_value(self) -> Union[None, float]:
        current_value = self.logger_manager["learner"]["steps"]
        return None if current_value.inner is None else current_value.inner.value  # type: ignore

    def wait(self) -> None:
        while True:
            current_value = self.get_current_value()
            if current_value is not None and current_value > self.num_steps:
                return
            time.sleep(1)
