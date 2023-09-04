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

ARG BASE_IMAGE=debian:bullseye-slim

FROM mambaorg/micromamba:1.4-bullseye as conda

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

## Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yaml /tmp/environment.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt

ARG CUDA_MAJOR=12
ARG ACCELERATOR
RUN echo "CUDA_MAJOR=${CUDA_MAJOR}"
RUN if [ "${ACCELERATOR}" = "cuda" ]; then \
      sed -i "s/jax==/jax[cuda${CUDA_MAJOR}_pip]==/g" /tmp/environment.yaml ; \
    elif [ "${ACCELERATOR}" = "tpu" ]; then \
      sed -i "s/jax==/jax[tpu]==/g" /tmp/environment.yaml ; \
      sed -i "s/jax_cuda_releases.html/libtpu_releases.html/g" /tmp/environment.yaml ; \
    elif [ "${ACCELERATOR}" = "cpu" ]; then \
      sed -i "s/jax==/jax[cpu]==/g" /tmp/environment.yaml ; \
    else \
    echo "ACCELERATOR should be cpu, cuda or tpu, but got: \"${ACCELERATOR}\"" && exit 1; \
    fi

RUN cat /tmp/environment.yaml

RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

FROM $BASE_IMAGE as test-image

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y curl

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV PATH=/opt/conda/envs/sebulba/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/opt/conda/envs/sebulba/lib

COPY . $APP_FOLDER

ARG NEPTUNE
RUN if [ "${NEPTUNE}" = "yes" ]; then \
  pip install --user -e ".[neptune]"; \
else \
  pip install --user -e .; \
fi
