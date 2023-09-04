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

import collections
import time
from typing import Any, Deque, Dict, List, Union

import numpy as np

from sebulba import core


class Counter:
    def __init__(self) -> None:
        self.value = 0
        self.queue: Deque = collections.deque(maxlen=5)
        self.flush_times: Deque = collections.deque(maxlen=5)

    def flush(self) -> Dict[str, float]:
        now = time.time()
        values = {
            "counter": float(self.value),
        }
        if len(self.queue) != 0:
            values["irate"] = (self.value - self.queue[-1]) / (
                now - self.flush_times[-1]
            )
            if len(self.queue) == 5:
                values["rate_5"] = (self.value - self.queue[0]) / (
                    now - self.flush_times[0]
                )

        self.queue.append(self.value)
        self.flush_times.append(now)
        return values

    def add(self, n: int) -> None:
        self.value += n


class MeanBetweenFlush:
    def __init__(self) -> None:
        self.values: List[float] = []

    def append(self, n: float) -> None:
        self.values.append(n)

    def flush(self) -> Dict[str, float]:
        values = self.values
        self.values = []
        r = {}
        if len(values) > 0:
            r["mean"] = float(np.mean(values))
            r["min"] = float(np.min(values))
            r["max"] = float(np.max(values))

        return r


class Recorder:
    def record_info(self, namespace: str, config: Dict[str, Any]) -> None:
        pass

    def record(self, metric: str, value: Any) -> None:
        pass

    def record_multiple(self, values: Dict[str, float]) -> None:
        pass

    def stop(self) -> None:
        pass


class HubItem:
    def __init__(self, parent: Union[None, "HubItem"] = None) -> None:
        self.inner: Union[None, Counter, MeanBetweenFlush] = None
        self.parent = parent

    def add(self, value: Any) -> None:
        if self.inner is None:
            if self.parent is not None:
                self.parent.add(value)
            self.inner = Counter()
            self.inner.add(value)
        elif not isinstance(self.inner, Counter):
            raise RuntimeError(
                f"This is not a counter: {self.inner.__class__} your can't use add"
            )
        elif self.parent is not None:
            self.parent.add(value)
        self.inner.add(value)

    def append(self, value: Any) -> None:
        if self.inner is None:
            if self.parent is not None:
                self.parent.append(value)
            self.inner = MeanBetweenFlush()
        elif not isinstance(self.inner, MeanBetweenFlush):
            raise RuntimeError(
                f"This is not a MeanBetweenFlush: {self.inner.__class__} your can't use append"
            )
        elif self.parent is not None:
            self.parent.append(value)
        self.inner.append(value)

    def create_sub(self) -> "HubItem":
        return HubItem(self)

    def flush(self) -> Dict[str, float]:
        if self.inner is None:
            return {}
        else:
            return self.inner.flush()


class Hub:
    def __init__(self, name: str, parent: Union[None, "Hub"] = None) -> None:
        self.name = name
        self.loggers: Dict[str, HubItem] = {}
        self.parent = parent
        self.childs: List["Hub"] = []

    def create_sub(self, name: str) -> "Hub":
        c = Hub(name, self)
        self.childs.append(c)
        return c

    def __getitem__(self, key: str) -> HubItem:
        if key in self.loggers:
            return self.loggers[key]
        if self.parent is not None:
            parrent_l = self.parent[key]
            logger = parrent_l.create_sub()
        else:
            logger = HubItem()
        self.loggers[key] = logger
        return logger

    def flush(self, details_level: int) -> Dict[str, float]:
        values = {}
        names = list(self.loggers.keys())
        for name in names:
            logger = self.loggers[name]
            for name2, value in logger.flush().items():
                values[f"{self.name}/{name}/{name2}"] = value
        return values


class LoggerManager(core.StoppableComponent):
    def __init__(self, recorder: Recorder, logger_flush_dt: float = 0.5) -> None:
        super(LoggerManager, self).__init__()
        self.recorder = recorder
        self.logger_flush_dt = logger_flush_dt
        self.loggers: Dict[str, Hub] = {}

    def __getitem__(self, name: str) -> Hub:
        if name in self.loggers:
            return self.loggers[name]
        c = Hub(name)
        self.loggers[name] = c
        return c

    def _run(self) -> None:
        before = time.time()
        while not self.should_stop:
            sleep_time = before - time.time() + self.logger_flush_dt
            if sleep_time > 0:
                time.sleep(sleep_time)
            before = time.time()
            self.flush()
            after = time.time()
            self.recorder.record("flush_time", after - before)
        self.recorder.stop()

    def flush(self) -> None:
        values = {}
        names = list(self.loggers.keys())
        for name in names:
            logger = self.loggers[name]
            values.update(logger.flush(0))
        self.recorder.record_multiple(values)


class RecordTimeTo:
    def __init__(self, to: HubItem):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)
