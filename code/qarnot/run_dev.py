#!/usr/bin/env python3

import qarnot
from qarnot.task import ForcedConstant, ForcedConstantAccess, ForcedNetworkRule

import os
from dotenv import load_dotenv
load_dotenv()

from qarnot.scheduling_type import OnDemandScheduling


# Registry settings
ACCOUNT="qarnotlab"
REPO="qarnotlab/blank-images"
TAG="qarnot_cuda-12.3.1-devel-ubuntu22.04"
PASSWORD= os.environ.get('DOCKER_HUB_PASSWORD')


# Tasq Settings
PROFILE="docker-nvidia-bi-a6000-network-ssh"
TASK_NAME=f"ssh qarnot_cuda-12.3.1-devel-ubuntu22.04"

conn = qarnot.connection.Connection(client_token='<<<MY_SECRET_TOKEN>>>')

# Use qarnotlab image
task = conn.create_task(TASK_NAME, PROFILE, 1)
task.constants["DOCKER_REPO"] = REPO
task.constants["DOCKER_TAG"] = TAG
task.constants["DOCKER_SRV"] = "https://registry-1.docker.io"

task.constants["DOCKER_REGISTRY_LOGIN"] = ACCOUNT
task.constants["QARNOT_SECRET__DOCKER_REGISTRY_PASSWORD"] = PASSWORD

# Setup SSH
task.constants['DOCKER_SSH'] = 'MY_SSH_KEY'

# CMD
task.constants['DOCKER_CMD'] = '/bin/bash -c "set_ssh && sleep infinity | tee master.logs"'


# On Demand - in order to not being preempted
task.scheduling_type=OnDemandScheduling()


# Buckets

# Create the input bucket and synchronize with a local folder
# Insert a local folder directory
input_bucket = conn.create_bucket("cuda-qarnot-in")
input_bucket.sync_directory("input")

# Attach the bucket to the task
task.resources.append(input_bucket)

# Append an output bucket
task.results = conn.create_bucket("cuda-qarnot-out")

# Define checkpoints in seconds
task.snapshot(600)

# Submit the task
task.submit()

# The following will print the state of the task to your console
# It will also print the command to connect through ssh to the task when it's ready
LAST_STATE = ''
SSH_TUNNELING_DONE = False
while not SSH_TUNNELING_DONE:
    if task.state != LAST_STATE:
        LAST_STATE = task.state
        print(f"** {LAST_STATE}")

    # Wait for the task to be FullyExecuting
    if task.state == 'FullyExecuting':
        # If the ssh connexion was not done yet and the list active_forward is available (len!=0)
        forward_list = task.status.running_instances_info.per_running_instance_info[0].active_forward
        if not SSH_TUNNELING_DONE and len(forward_list) != 0:
            ssh_forward_port = forward_list[0].forwarder_port
            ssh_forward_host = forward_list[0].forwarder_host
            cmd = f"ssh -o StrictHostKeyChecking=no root@{ssh_forward_host} -p {ssh_forward_port}"
            print(cmd)
            SSH_TUNNELING_DONE = True

    # Display errors on failure
    if task.state == 'Failure':
        print(f"** Errors: {task.errors[0]}")
        SSH_TUNNELING_DONE = True