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

import os
import pickle

from sebulba import types


class SyncSaveParams:
    def __init__(self, path: str, save_every: int):
        self.path = path
        self.iteration = 0
        self.save_every = save_every
        if not os.path.exists(path):
            os.makedirs(path)

    def __call__(self, new_params: types.Params) -> None:
        if self.iteration % self.save_every == 0:
            with open(
                os.path.join(self.path, f"{self.iteration}_params.pickle"), "wb"
            ) as f:
                pickle.dump(new_params, f)
        self.iteration += 1
