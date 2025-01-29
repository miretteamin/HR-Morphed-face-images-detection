import qarnot
from qarnot.task import ForcedConstant
import os
import time
import json
from dotenv import load_dotenv
from qarnot.scheduling_type import OnDemandScheduling

load_dotenv()

with open("code\qarnot\launch_model_qarnot.json", "r") as config_file:
    config = json.load(config_file)

PROFILE = config["QARNOT_PROFILE"]
TASK_NAME = config["QARNOT_TASK_NAME"]

print(f"Creating task {TASK_NAME} with profile {PROFILE}")
conn = qarnot.connection.Connection(client_token=config.get("QARNOT_TOKEN"))

# Create task
task = conn.create_task(TASK_NAME, PROFILE, 1)
task.constants["DOCKER_REPO"] = config["QARNOT_REPO"]
task.constants["DOCKER_TAG"] = config["QARNOT_TAG"]
task.constants["DOCKER_SRV"] = config["DOCKER_SRV"]

task.constants["DOCKER_REGISTRY_LOGIN"] = config["DOCKER_ACCOUNT"]
task.constants["QARNOT_SECRET__DOCKER_REGISTRY_PASSWORD"] = config["DOCKER_PASSWORD"]

task.constants['DOCKER_SSH'] = config["DOCKER_SSH"]

# Set hardware requirements
task.resources_constraints = {"cpu_count": {"min": config["QARNOT_CPU_MIN"]},}

task.constants['DOCKER_CMD'] = f'/bin/bash -c " \
    echo \"🔑 Ensuring SSH key exists inside the container...\" && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    echo \"{task.constants["DOCKER_SSH"]}\" > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    echo \"✅ SSH Key added to authorized_keys!\" && \
    set_ssh && sleep infinity | tee master.logs & \
    nvidia-smi && \
    apt-get update -y && \
    apt-get install apt-utils -y && \
    apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /job/code/requirements.txt && \
    python3 -u /job/code/run.py train --config /job/code/configs/config_memmap.json --datadir /job/ --memmap True"'

# task.constants['DOCKER_CMD'] = config["DOCKER_CMD"].replace("{DOCKER_SSH}", task.constants["DOCKER_SSH"])


# Retrieve buckets
input_bucket = conn.retrieve_bucket(config["INPUT_BUCKET"])
output_bucket = conn.retrieve_bucket(config["OUTPUT_BUCKET"])

task.resources.append(input_bucket)
task.results = output_bucket  

task.snapshot(600)  # Define checkpoints in seconds

# Submit task
try:
    task.submit()
except Exception as e:
    print(f"❌ ERROR: Task submission failed: {e}")
    exit(1)

print("🚀 Task submitted! Waiting for execution...\n")

LAST_STATE = ''
SSH_TUNNELING_DONE = False

while not SSH_TUNNELING_DONE:
    logs = task.stdout().split("\n")  # Get logs
    if task.state != LAST_STATE:
        LAST_STATE = task.state
        print(f"\n📌 Current Task State: {LAST_STATE}")

    print(f"\n🔍 Checking task state: {task.state} ...")
    print("\n📝 Task Logs:")
    for line in logs:
        print(f"   {line}")

    if task.state == 'FullyExecuting':
        print("\n✅ Task is now executing!")
        forward_list = task.status.running_instances_info.per_running_instance_info[0].active_forward
        if not SSH_TUNNELING_DONE and len(forward_list) != 0:
            ssh_forward_port = forward_list[0].forwarder_port
            ssh_forward_host = forward_list[0].forwarder_host
            cmd = f"ssh -o StrictHostKeyChecking=no root@{ssh_forward_host} -p {ssh_forward_port}"
            print(f"\n🛠️ SSH Command to connect: {cmd}")
            SSH_TUNNELING_DONE = True

    if task.state == 'Failure':
        print(f"\n❌ Task failed with errors: {task.errors}")
        SSH_TUNNELING_DONE = True

    time.sleep(10)
