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

from typing import Any, Dict, List

import neptune

from sebulba.logging import core


class NeptuneRecorder(core.Recorder):
    def __init__(self, name: str, tags: List[str], project: str):
        self.run = neptune.init_run(
            project=project,
            name=name,
            git_ref=neptune.types.atoms.git_ref.WithDisabledMixin.DISABLED,
            tags=list(tags),
        )

    def record_info(self, namespace: str, info: Any) -> None:
        self.run[namespace] = info

    def record(self, metrics: str, value: Any) -> None:
        self.run[metrics].append(value)

    def record_multiple(self, values: Dict[str, float]) -> None:
        if len(values) != 0:
            self.run[""].append(values)

    def stop(self) -> None:
        self.run.stop()
