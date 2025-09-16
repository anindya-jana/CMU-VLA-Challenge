# CMU-VLA-Challenge(This is the same as challange)
## To view the updates for the Challenge view the other branch.

## Table of Contents
[Introduction](#introduction)  
[Objective](#objective)  
[Task Specification](#task-specification)

[Setting Up](#setting-up)
- [Challenge Scenes](#challenge-scenes)

[Real-Robot Challenge](#real-robot-challenge-2025)
- [Real-Robot Data](#real-robot-data)

[Submission](#submission)

[Evaluation](#evaluation)
- [Question Types and Initial Scoring](#question-types-and-initial-scoring)
- [Timing](#timing)

[Challenge FAQ](#challenge-faq)

## Introduction
The CMU Vision-Language-Autonomy Challenge leverages computer vision and natural language understanding in navigation autonomy. The challenge aims at pushing the limit of embodied AI in real environments and on real robots - providing a robot platform and a working autonomy system to bring everybody's work a step closer to real-world deployment. The challenge provides a real-robot system equipped with a 3D lidar and a 360 camera. The system has base autonomy onboard that can estimate the sensor pose, analyze the terrain, avoid collisions, and navigate to waypoints. Teams will set up software on the robot's onboard computer to interface with the system and navigate the robot. For 2024, the challenge will be done in a custom simulation environment and move to the real-robot system the following year. 

To register for the challenge, please see our [Challenge Website](https://www.ai-meets-autonomy.com/cmu-vla-challenge).


## Objective 
Teams are expected to come up with a vision-language model that can take a natural language navigation query and navigate the vehicle system by generating a waypoint or path based on the query.


## Task Specification
In the challenge, teams are provided with a set of natural language questions/statements for scenes from Unity [1]. The team is responsible for developing software that processes the questions together with onboard data of the scene provided by the system. The questions/statements all contain a spatial reasoning component that requires semantic spatial understanding of the objects in the scene. The environment is initially unknown and the scene data is gathered by navigating to appropriate viewpoints and exploring the scene by sending waypoints to the system. 5 questions/statements are provided for each of 15 Unity scenes and 3 scenes are held out for test evaluation.

The natural language questions are separated into three categories: numerical, object reference, and instruction following, which are further described below.

**Numerical**

Numerical questions asks about the quantity of an object that fits certain attributes or spatial relations. The response is expected to be an integer number.

Examples:

    How many blue chairs are between the table and the wall?

    How many black trash cans are near the window? 

**Object Reference**

Object reference statements asks the system to find a certain object located in the scene that is referred to by spatial relations and/or attributes. The response is expected to be a bounding box around the object and there exists only one correct answer in the scene (the referred object is unique). The center point of the bounding box marker will be used as a waypoint to navigate the robot system.

Examples:

    Find the potted plant on the kitchen island that is closest to the fridge.

    Find the orange chair between the table and sink that is closest to the window.

**Instruction-Following**

Instruction following statements ask the system to take a certain path, using objects to specify the trajectory of the path. The response is expected to be a sequence of waypoints.

Examples:

    Take the path near the window to the fridge.

    Avoid the path between the two tables and go near the blue trash can near the window.


## Setting Up
First, clone this repo and place it under your local `/home/$USER/` folder.

### Challenge Scenes
A total of 18 Unity scenes are used for the challenge. 15 scenes are provided for model development while 3 are held out for testing. The majority of these scenes are single rooms while a few are multi-room buildings.  A set of the training environment models can be downloaded from [here](https://drive.google.com/drive/folders/1bmxdT6Oxzt0_0tohye2br7gqTnkMaq20?usp=share_link). For all of the 15 training scenes, we also provide a processed point cloud of the scene, object and region information including color and size attributes, and referential language statements (please see [Object-Referential Dataset](#object-referential-dataset-vla-3d) for more details). 

![image](figures/scenes.png)


## Real-Robot Challenge (2025)

Starting in 2025, the final round of challenge evaluation will be done on the real-robot system while initial evaluation rounds are still done in simulation. Similar to the simulator, the system provides onboard data as described below and takes waypoints in the same way as the simulator. The software developed in the AI module is only able to send waypoints to explore the scene. Manually sending waypoints or teleoperation is not allowed. During the final evaluation phase, each team will remotely login to the robot's onboard computer (16x i9 CPU cores, 32GB RAM, RTX 4090 GPU), and set up software in a Docker container that interfaces with the autonomy modules. The Docker container is used by each team alone and not shared with other teams. We will schedule time slots for teams who pass the simulation round to set up the software and test the robot during that phase. The teams can also record data on the robot's onboard computer and this data will be made available to participants afterwards.

### Real-Robot Data

Example scene data collected from the real system is provided [here](https://drive.google.com/drive/folders/1M0_UkY7aDIEpoVK6GdWzg45v-HX2UMSd?usp=drive_link) with some differences in the object layout. The following can be found in the sample data:

- `data_view.rviz`: An RVIZ configuration file provided for viewing the data
- `map.ply`: A ground truth map with object segmentation and IDs
- `object_list.txt`: Object list with bounding boxes and labels are also provided
- `system.zip`: Zipped bagfile with ROS messages provided by the system in the same format as during the challenge
- `readme.txt`: Calibration information and further details about the sample files

Here, the ground truth map and the object list are not provided files during the challenge but shown as a sample of what information can be obtained and processed from the system. The camera pose (camera frame) with respect to the lidar (sensor frame) can be found in the README file included. Further details about the files can be found in the README text file as well.


## Submission
Submissions will be made as a github repository link to a public repository. The easiest way would be to fork this repository and make changes there, as the repository submitted will need to be structured in the same way. The only files/folders that should be changed are what's under [ai_module](ai_module/) and potentially the [launch.sh](launch.sh). If changes were made to the docker image to install packages, push the updated image to [Docker Hub](https://hub.docker.com/) and submit the link to the image as well.

Prior to submitting, please download the docker image and test it with the simulator as the submission will be evaluated in the same way. Please also make sure that waypoints and visualization markers sent match the types in the example dummy model and are on the same ROS topics so that the base navigation system can correctly receive them.

Please fill out the [Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSfcKWEV3ReuGEfGag0En706KDtgxjFDayE6dJIgWElQrXiDmw/viewform?usp=sharing&ouid=113649899278879140488) with a link to your Github repo.


## Evaluation
The submitted code will be pulled and evaluated with 3 Unity environment models which have been held from the released data. Each scene will be unknown and the module has a set amount of time to explore and answer the question (see [timing](#timing) for more details). The test scenes are of similar style to the provided training scenes. **The system will be relaunched for each language command tested such that information collected from previously exploring the scene is not retained.** Note that the information onboard the system that is allowed to be used at test time is limited to what is listed in [System Outputs](#system-outputs).

Evaluation is performed by a `challenge_evaluation_node` whose source code is not made public. The evaluation node will be started along with the team-provided AI module and the system at the same time, and publishes a single question each startup as a ROS String message on the following topic at a rate of 1Hz:

| Message | Description | Frequency | ROS Topic Name |
|-|-|-|-|
| Challenge Question | ROS Pose2D message with position and orientation. | 1Hz | `/challenge_question` |

### Question Types and Initial Scoring

For each scene, 5 questions similar to those provided will be tested and a score will be given to each response. The question types will be scored as follows:
- **Numerical** (/1): Exact number must be published on `/numerical_response` as an `std_msgs/Int32` message. Score of 0 or 1.
- **Object Reference** (/2): ROS `visualization_msgs/Marker` message must be published on `/selected_object_marker`, and is scored based on its degree of overlap with the ground truth object bounding box. Score between 0 and 2.
- **Instruction-Following** (/6): A series of `geometry_msgs/Pose2D` waypoints must be published on `/way_point_with_heading` to guide the vehicle. The score will be calculated based on the actual trajectory followed by the robot based on whether it follows the path constraints in the command and in the correct order. Penalties are imposed upon the score if the followed path deviates from the correct order of constraints, does not achieve the desired constraints, or passes through areas it is forbidden to go through in the command. Score between 0 and 6, with possibility for partial points. 

The scores from all questions across the 3 test scenes will be totaled for each team's final score. 

Note: Teams have a choice whether to use the ground-truth semantics posted on the /object_markers topic in their final submission. Methods that do not use the published ground-truth semantics will be scored differently.

### Timing

For each question, both re-exploration on system launch and question answering will be timed. Timing will begin immediately at system startup. Each question has a total time limit of **10 minutes** for exploration and question answering combined, regardless of the test scene. Exceeding the time limit for a certain question incurs a penalty on the initial score calculated for the question. Finishing before the allotted time for a question earns bonus points on that question, which will be used to break ties between teams with similar initial scores.




## Acknowledgements
Thank you to [AlphaZ](https://alpha-z.ai/) for sponsoring the challenge for 2025! Their generous support enables us to provide the top three teams with a cash prize.

## References

[1] J. Haas. "A history of the unity game engine," in Diss. Worcester Polytechnic Institute, vol. 483, no. 2014, pp. 484, 2014.
