
# CMU VLA Challenge Installation Instructions

First clone the repo in /home/${USER}:
```
git clone -b ai-module-updates --single-branch https://github.com/anindya-jana/CMU-VLA-Challenge.git
```

<img width="1190" height="1345" alt="Screenshot from 2025-09-16 15-43-15" src="https://github.com/user-attachments/assets/ac359864-57e8-4cd7-a3bf-b20fa317b1f4" />

# CMU VLA Challenge Podman Instructions
Two docker images are used for the challenge:
- `ubuntu20_ros_system`: docker image for the system simulator - this image should NOT be modified
- `ubuntu20_ros`: docker image for the AI module - this will be the image you modify when developing the model
  
## 1) Allow X11 and ensure Podman runtime dir
```
xhost +
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p "$XDG_RUNTIME_DIR/containers"
chmod 700 "$XDG_RUNTIME_DIR"
```
## 2) Start containers 

Option A (native):
```
podman compose -f docker/podman-compose.gpu.yml up -d
```
Option B (if native compose unavailable):
```
podman-compose -f docker/podman-compose.gpu.yml up -d
```
## 3) Ensure both container are running:
```
podman start ubuntu20_ros
```
```
podman start ubuntu20_ros_system
```
then go to the respective folder and then 
```
catkin make
```
## 4) Launch the simulator (runs Unity + roslaunch):
```
podman exec -it ubuntu20_ros_system bash -lc "cd /home/${USER}/CMU-VLA-Challenge && ./launch_system.sh"
```
   Notes:
   - If you see missing PCL/cv_bridge or image_transport/rviz/tf binaries after a reboot, install once:
```
podman exec -it ubuntu20_ros_system bash -lc "export DEBIAN_FRONTEND=noninteractive; apt-get update -y; apt-get install -y --no-install-recommends libpcl-dev pcl-tools ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-image-transport-plugins ros-noetic-rviz ros-noetic-tf ros-noetic-diagnostic-updater ros-noetic-diagnostic-aggregator libopencv-dev; ldconfig; [ -e /usr/bin/python ] || ln -s /usr/bin/python3 /usr/bin/python"
```
   - Then run.
```
bash -lc 'set -e; echo "== Podman runtime repair =="; export XDG_RUNTIME_DIR=/run/user/$(id -u); mkdir -p "$XDG_RUNTIME_DIR/containers"; podman system migrate || true; xhost + &>/dev/null || true; if podman compose version &>/dev/null; then COMPOSE="podman compose -f docker/podman-compose.gpu.yml"; else COMPOSE="podman-compose -f docker/podman-compose.gpu.yml"; fi; echo "== Compose down =="; $COMPOSE down --remove-orphans || true; echo "== Remove old containers =="; podman rm -f ubuntu20_ros_system ubuntu20_ros &>/dev/null || true; echo "== Compose up =="; $COMPOSE up -d; echo "== Status after up =="; podman ps --format "{{.Names}} {{.Status}}"; echo "== Launching SYSTEM (Unity/RViz) =="; podman exec -it ubuntu20_ros_system bash -lc '"'"'set -e; source /opt/ros/noetic/setup.bash; cd /home/'"${USER}"'/CMU-VLA-Challenge; rm -f /root/system_launch.log; nohup ./system_bring_up.sh >/root/system_launch.log 2>&1 & disown; echo SYSTEM_STARTED'"'"'; echo "== Launching AI module =="; podman exec -it ubuntu20_ros bash -lc '"'"'set -e; source /opt/ros/noetic/setup.bash; if [ -f /home/'"${USER}"'/CMU-VLA-Challenge/ai_module/devel/setup.bash ]; then source /home/'"${USER}"'/CMU-VLA-Challenge/ai_module/devel/setup.bash; fi; rm -f /root/ai_launch.log; nohup roslaunch dummy_vlm dummy_vlm.launch >/root/ai_launch.log 2>&1 & disown; echo AI_STARTED'"'"'; sleep 2; echo "== Final status =="; podman ps --format "{{.Names}} {{.Status}}"; echo "--- System log (tail) ---"; podman exec -it ubuntu20_ros_system bash -lc '"'"'for i in $(seq 1 20); do [ -s /root/system_launch.log ] && break; sleep 1; done; tail -n 120 /root/system_launch.log || true'"'"'; echo "--- AI log (tail) ---"; podman exec -it ubuntu20_ros bash -lc '"'"'for i in $(seq 1 20); do [ -s /root/ai_launch.log ] && break; sleep 1; done; tail -n 120 /root/ai_launch.log || true'"'"''

```
   - Then re-run the launch command above.

## 4) To install owlvit and other dependencies on ai container:
```
podman exec -it ubuntu20_ros bash -lc "
  # Update package lists and install python3 and pip if not already present
  apt update && \
  apt install -y python3 python3-pip && \
  
  # Install core Python libraries (torch for CPU, transformers, pillow, safetensors)
  # Using --no-cache-dir to prevent filling up container disk with pip cache
  pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
  pip install --no-cache-dir transformers Pillow safetensors && \
  
  # Execute the Python script to download and cache the model
  python3 -c \"
import sys
try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    print('Transformers library loaded successfully.')
except ImportError:
    # Use double quotes for the Python string, and include the single quotes directly
    print('Error: transformers library not found. Please install it with \\'pip install transformers\\'.', file=sys.stderr)
    sys.exit(1)

try:
    print('Attempting to download OwlViT model and processor...')
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')
    print('Model and processor for google/owlvit-base-patch32 have been downloaded and cached.')
except Exception as e:
    # f-string using single quotes is clean
    print(f'An error occurred during model download: {e}', file=sys.stderr)
    sys.exit(1)
\"
"
```
 
## 5)  Launch the AI module (web UI + ROS node):
```
podman exec -it ubuntu20_ros bash -lc "export MPLBACKEND=Agg TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0; [ -e /usr/bin/python ] || ln -s /usr/bin/python3 /usr/bin/python; source /opt/ros/noetic/setup.bash; source /home/${USER}/CMU-VLA-Challenge/ai_module/devel/setup.bash; roslaunch dummy_vlm dummy_vlm.launch"
```
Open the UI in a browser:
   http://localhost:16552



# CMU VLA Challenge Docker Instructions(as given in CMU Challenge)
Two docker images are used for the challenge:
- `ubuntu20_ros_system`: docker image for the system simulator - this image should NOT be modified
- `ubuntu20_ros`: docker image for the AI module - this will be the image you modify when developing the model

You may modify `Dockerfile` to edit the docker image for the module developed. 

Prior to following these instructions, make sure you have pulled this repo and copied it to your `/home/$USER` folder. If you want to use a different path, refer to the note under the section [Run and Modify Docker Image](#run-and-modify-docker-image).

## Install Docker

### 1) For computers without a Nvidia GPU

Install Docker and grant user permission:
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install additional packages:
```
sudo apt update && sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

### 2) For computers with Nvidia GPUs

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install Nvidia Container Toolkit (Nvidia GPU Driver
should be installed already).

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```
sudo apt update && sudo apt install nvidia-container-toolkit
```
Configure Docker runtime and restart Docker daemon.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Test if the installation is successful, you should see something like below.
```
docker run --gpus all --rm nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```
```
Sat Dec 16 17:27:17 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 24%   50C    P0    40W / 200W |    918MiB /  8192MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## Run and Modify Docker Image
Move the entire `CMU-VLA-Challenge` repo to your local `/home/$USER` folder so that it is the working directory in the docker image.

Note: If you choose to place it elsewhere, you'll have to modify the `working_dir` parameter in the docker compose file(s) to match the path to your cloned repo.

Inside the `CMU-VLA-Challenge` folder, allow remote X connection:
```
xhost +
```
Go inside this folder in terminal.
```
cd CMU-VLA-Challenge/docker/
```

For computers **without a Nvidia GPU**, compose the Docker image and start the containers:
```
docker compose -f compose.yml up --build -d
```
For computers **with Nvidia GPUs**, use the `compose_gpu.yml` file instead (creating the same Docker image, but starting the container with GPU access):
```
docker compose -f compose_gpu.yml up --build -d
```
This will start two docker containers. One will be for the challenge simulator system and the other will be the development docker for the AI module which you will modify.

Access the running containers:
```
docker exec -it ubuntu20_ros_system bash
docker exec -it ubuntu20_ros bash
```

## Set Up and Launch Entire System
Set up the simulator with Unity environment models. The simulator can also be launched by itself - more details can be found in [system/unity](system/unity).

Install dependencies with the command lines below:
```
sudo apt update
sudo apt install libusb-dev python-yaml python-is-python3
```
In a terminal, go inside the [system/unity](../system/unity) folder and compile (this may take a few minutes):
```
catkin_make
```
Download any of our [Unity environment models](https://drive.google.com/drive/folders/1bmxdT6Oxzt0_0tohye2br7gqTnkMaq20?usp=share_link), unzip the folder, and copy the files inside to the [system/unity/src/vehicle_simulator/mesh/unity](../system/unity/src/vehicle_simulator/mesh/unity/) folder. The environment model files should follow the structure below. Note that the `AssetList.csv` file is generated upon start of the system and that only one given environment folder can be placed under the [system/unity/src/vehicle_simulator/mesh/unity](../system/unity/src/vehicle_simulator/mesh/unity/) directory at a time.

mesh/<br>
&nbsp;&nbsp;&nbsp;&nbsp;unity/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;environment/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model_Data/ (multiple files in the folder)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model.x86_64<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UnityPlayer.so<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AssetList.csv (generated at runtime)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dimensions.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Categories.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object_list.txt<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;traversable_area.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;render.jpg<br>

To test whether the simulator was correctly set up, go inside [system/unity](../system/unity) and run:
```
./system_bring_up.sh
```
Go inside the [ai_module](../ai_module/) folder and compile and set up the package:
```
catkin_make
```
Inside the docker for the simulator system, the system can be launched under the root repository directory with:
```
./launch_system.sh
```
Inside the docker for the AI module, the dummy model can be launched under the root repository directory with:
```
./launch_module.sh
```
You should see both the simulator launching in RViz in one docker and a terminal prompt asking for text input in the other.

The prompt will ask you to type in a question or command and the system will move accordingly. As the system is running with a "dummy model" by default, it simply parses the type of statement and returns the appropriate response type with arbitrary values. The behavior of the dummy model for different language inputs is as follows: 
- "how many...": prints out a number in terminal
- "find the...": highlights the object with a visualization marker and navigates to it
- anything else: sends a series of fixed waypoints

If you use the control panel to navigate the vehicle, to resume waypoint navigation afterwards, click the 'Resume Navigation to Goal' button. The contents under the [ai_module](../ai_module) folder can be modified and the [dummy_vlm](../ai_module/src/dummy_vlm/) package replaced with yours.
