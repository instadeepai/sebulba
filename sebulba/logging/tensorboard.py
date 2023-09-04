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

import re
import time
from os import path
from typing import Any, Dict, List

import tensorboardX

from sebulba import logging


class TensorboardRecorder(logging.Recorder):
    def __init__(self, name: str, tags: List[str], logdir: str = "./logs"):
        logdir_name = re.sub("[^\\w\\s-]", "", name).strip().lower()
        logdir_name = re.sub("[-\\s]+", "-", logdir_name) + "-" + str(int(time.time()))
        self.writer = tensorboardX.SummaryWriter(
            path.join(logdir, logdir_name),
            name,
        )
        self.writer.add_text("tag", str(tags))
        self.step = 0

    def record_info(self, key: str, info: Dict[str, Any]) -> None:
        self.writer.add_text(key, str(info))

    def record(self, metric: str, value: Any) -> None:
        self.writer.add_scalar(metric, value, self.step)

    def record_multiple(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.record(k, v)
        self.step += 1

    def stop(self) -> None:
        self.writer.flush()
