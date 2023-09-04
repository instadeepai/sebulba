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

import multiprocessing
import platform
from typing import Dict

import requests
import yaml

GCP_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance"


def get_vm_attribute(attribute_name: str) -> str:
    return requests.get(
        f"{GCP_METADATA_URL}/attributes/{attribute_name}",
        headers={"Metadata-Flavor": "Google"},
    ).text


def get_tpu_info() -> Dict[str, str]:
    try:
        env = yaml.load(get_vm_attribute("tpu-env"), yaml.SafeLoader)
        return {
            "type": env["TYPE"],
            "accelerator-type": env["ACCELERATOR_TYPE"],
            "topology": env["TOPOLOGY"],
            "zone": env["ZONE"],
            "project-id": env["CONSUMER_PROJECT_ID"],
        }
    except Exception:
        return {"fail-to-get-info": "True"}


def get_info() -> Dict[str, str]:
    infos = {}

    infos.update(get_tpu_info())

    infos["cpus"] = str(multiprocessing.cpu_count())
    infos["cpu-type"] = platform.machine()
    infos["hostname"] = platform.node()

    return infos
