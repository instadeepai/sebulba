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

#################
# General setup #
#################

GIT_REPO?=instadeepai/sebulba
GIT_BRANCH?=main
CHECKOUT_DIR?=sebulba


#######
# TPU #
#######
# Shared set-up.
ENV_CONFIG=CLOUDSDK_COMPUTE_ZONE=$(ZONE) CLOUDSDK_CORE_PROJECT=$(PROJECT)
BASE_CMD=${ENV_CONFIG} gcloud alpha compute tpus tpu-vm
BASE_CMD_QUEUED=${ENV_CONFIG} gcloud alpha compute tpus queued-resources
BASE_NAME?=sebulba
WORKER=all

# Basic TPU configuration.
PROJECT?=your-gcp-project
ZONE?=us-central2-b
ACCELERATOR_TYPE?=v4-8
RUNTIME_VERSION?=tpu-vm-v4-base
NAME?=$(BASE_NAME)-$(ACCELERATOR_TYPE)

info:
	@echo BASE_NAME=$(BASE_NAME)
	@echo PROJECT=$(PROJECT)
	@echo ZONE=$(ZONE)
	@echo ACCELERATOR_TYPE=$(ACCELERATOR_TYPE)
	@echo RUNTIME_VERSION=$(RUNTIME_VERSION)

.PHONY: create_vm
create_vm:
	$(BASE_CMD) create $(NAME) \
		--accelerator-type $(ACCELERATOR_TYPE) \
		--version $(RUNTIME_VERSION) \

.PHONY: create
create: create_vm prepare_vm

.PHONY: git_clone
git_clone:
	$(BASE_CMD) ssh $(NAME)  \
		--command="GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git clone -b ${GIT_BRANCH} git@github.com:${GIT_REPO}.git ${CHECKOUT_DIR}" \
		-- -A

.PHONY:
git_pull:
	$(BASE_CMD) ssh $(NAME)  \
		--command="cd ${CHECKOUT_DIR} && git pull" \
		-- -A

.PHONY: connect
connect:
	$(BASE_CMD) ssh $(NAME)

.PHONY: list
list:
	$(BASE_CMD) list

VERBS = start describe stop delete

.PHONY: $(VERBS)
$(VERBS):
	$(BASE_CMD) $@ $(NAME)

.PHONY: delete_q
delete_q:
	$(BASE_CMD_QUEUED) delete \
		$(NAME)-q

.PHONY: run
run:
	$(BASE_CMD) ssh $(NAME) --worker=$(WORKER) --command="$(command)"

##########
# Docker #
##########

SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
NEPTUNE ?= no
NEPTUNE_API_TOKEN ?=
DOCKER_BASE_IMAGE = debian:bullseye-slim
DOCKER_BUILD_ARGS =

EXPERIMENT_NAME ?= tpu-v4-max-fps
USE_ONLY_NUMA_NODE0 ?= yes
ifeq ($(USE_ONLY_NUMA_NODE0), yes)
	DOCKER_RUN_FLAGS = --rm --privileged --network host --cpuset-cpus $(shell cat /sys/devices/system/node/node0/cpulist)
else
	DOCKER_RUN_FLAGS = --rm --privileged --network host
endif
DOCKER_VARS_TO_PASS ?=
DOCKER_IMAGE_NAME = sebulba_image
DOCKER_CONTAINER_NAME = sebulba_container
CUDA_MAJOR ?= 12


.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".pytest_cache" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

.PHONY: docker_build
docker_build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) --build-arg CUDA_MAJOR=$(CUDA_MAJOR) --build-arg ACCELERATOR=$(ACCELERATOR) --build-arg NEPTUNE=$(NEPTUNE) -f docker/Dockerfile .

.PHONY: docker_build_tpu
docker_build_tpu:
	$(MAKE) docker_build ACCELERATOR=tpu

.PHONY: docker_build_cpu
docker_build_cpu:
	$(MAKE) docker_build ACCELERATOR=cpu

.PHONY: docker_build_cuda
docker_build_cuda:
	$(MAKE) docker_build ACCELERATOR=cuda

.PHONY: docker_run
docker_run:
	sudo docker run $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) $(command)

.PHONY: docker_start
docker_start:
	sudo docker run -itd $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME)

.PHONY: docker_kill
docker_kill:
	sudo docker kill $(DOCKER_CONTAINER_NAME)

BASE_CMD_SSH_ALL = $(BASE_CMD) ssh $(NAME) --worker=all

.PHONY: remote_build_tpu
remote_build_tpu:
	$(BASE_CMD_SSH_ALL) --command="cd $(CHECKOUT_DIR); make NEPTUNE=$(NEPTUNE) docker_build_tpu"

.PHONY: docker_enter
docker_enter:
	sudo docker run $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app -it $(DOCKER_IMAGE_NAME)

.PHONY: remote_kill
remote_kill:
	$(BASE_CMD_SSH_ALL) --command="cd $(CHECKOUT_DIR); make docker_kill" || echo "No container to kill"

.PHONY: setup
setup: git_clone remote_build_tpu

.PHONY: run_script
run_script:
	$(BASE_CMD_SSH_ALL) --command="cd $(CHECKOUT_DIR); make docker_run USE_ONLY_NUMA_NODE0=$(USE_ONLY_NUMA_NODE0) DOCKER_VARS_TO_PASS='-e ACCELERATOR_TYPE=$(ACCELERATOR_TYPE) -e NEPTUNE_API_TOKEN=$(NEPTUNE_API_TOKEN)' command='python experiments/sebulba_ppo_atari.py +experiment=$(EXPERIMENT_NAME)'"

.PHONY: kill_pull_run
kill_pull_run: remote_kill git_pull run_script

BASE_CMD_SSH_HEAD = $(BASE_CMD) ssh $(NAME) --worker=0

.PHONY: start_tensorboard
start_tensorboard:
	$(BASE_CMD_SSH_HEAD) --command="sudo docker run -p 6006:6006 -d -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python -v /home/$$USERNAME/$(BASE_NAME)/logs/:/logs tensorflow/tensorflow tensorboard --logdir=/logs --bind_all"

.PHONY: port_forward_tensorboard
port_forward_tensorboard:
	$(BASE_CMD_SSH_HEAD) --ssh-flag="-C -N -L 6006:localhost:6006"
