import qarnot
from qarnot.task import ForcedConstant
import os
import time
import json
from dotenv import load_dotenv
from qarnot.scheduling_type import OnDemandScheduling

load_dotenv()

with open("qarnot\launch_model_qarnot.json", "r") as config_file:
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

## RUN 1
# task.constants['DOCKER_CMD'] = f'/bin/bash -c "set_ssh && \
#     mkdir -p /job/output/logs && \
#     nvidia-smi && \
#     apt-get update -y && apt-get install -y apt-utils python3 python3-pip openssh-client && \
#     python3 -m pip install --upgrade pip && \
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
#     pip install -r /job/code/requirements.txt && \
#     python3 -u /job/code/run.py train --config /job/code/configs/config_memmap.json --datadir /job/ --memmap True --weights /job/output/logs/resnet34_memmap_full-train_augment_no-downscale_lr_plateau/model_epoch_10.pth --wandb_run_id i3eq22bw && \
#     cp -r /job/output/* /qarnot-output/ && \
#     sleep 10"'

# ## RUN 2
task.constants['DOCKER_CMD'] = f'/bin/bash -c "set_ssh && \
    mkdir -p /job/output/logs && \
    nvidia-smi && \
    apt-get update -y && apt-get install -y apt-utils python3 python3-pip openssh-client && \
    python3 -m pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /job/code/requirements.txt && \
    python3 -u /job/code/run_2.py train --config /job/code/configs/config_memmap_2.json --datadir /job/ --memmap True --weights /job/output/logs/efficientnet_b0_memmap_full-train_augment_no-downscale_lr_plateau/model_epoch_10.pth --wandb_run_id 5e9fipbl && \
    cp -r /job/output/* /qarnot-output/ && \
    sleep 10"'

# task.constants['DOCKER_CMD'] = f'/bin/bash -c "set_ssh && \
#     mkdir -p /job/output/logs && \
#     echo \'🔍 Listing contents of /job/ to check if model is uploaded:\' && \
#     ls -R /job/ && \
#     echo \'🔍 Listing contents of /job/output/logs before execution:\' && \
#     ls -R /job/output/logs && \
#     echo \'🔍 Checking if model file exists:\' && \
#     if [ -f /job/output/logs/resnet34_memmap_full-train_augment_no-downscale_lr_plateau/model_epoch_10.pth ]; then \
#         echo \'✅ Model checkpoint found!\'; \
#     else \
#         echo \'❌ Model checkpoint NOT found! Check if it was uploaded correctly.\'; \
#     fi && \
#     echo \'🚀 Proceeding with Training Script Execution...\' && \
#     python3 -u /job/code/run.py train --config /job/code/configs/config_memmap.json --datadir /job/ --memmap True --weights /job/output/logs/resnet34_memmap_full-train_augment_no-downscale_lr_plateau/model_epoch_10.pth --wandb_run_id i3eq22bw && \
#     cp -r /job/output/* /qarnot-output/ && \
#     sleep 10"'


## there is a problem with echo, it assigns completed for the task without even being executed

# task.constants['DOCKER_CMD'] = config["DOCKER_CMD"].replace("{DOCKER_SSH}", task.constants["DOCKER_SSH"])


# Retrieve buckets
input_bucket1 = conn.retrieve_bucket(config["INPUT_BUCKET"])
input_bucket2= conn.retrieve_bucket(config["OUTPUT_BUCKET"])
output_bucket = conn.retrieve_bucket(config["OUTPUT_BUCKET"])

task.resources.append(input_bucket1)
task.resources.append(input_bucket2)
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
