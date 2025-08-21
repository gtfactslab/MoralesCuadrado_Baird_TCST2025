# Code for the TCST 2025 Submission: "Lightweight Tracking Control for Computationally Constrained Aerial Systems with Newton-Raphson Flow"
Video footage for certain flights may be seen [here](https://youtu.be/H0mwDMPMdxQ)
# Environment Setup
1. Users of this code must use the `/src/` folder here in a ROS2 workspace and build
2. Users may also use the environment.yml file to set up a conda environment with the necessary packages to run the code below

# Quadrotor Code Instructions
## Preliminary
1. Follow the instructions ([here](https://docs.px4.io/main/en/ros/ros2_comm.html)) to set up the PX4 Autopilot Stack, ROS2, Micro XRCE-DDS Agent & Client, and build a ROS2 workspace with the necessary px4 communication repos
2. In the same workspace with the communication folders as above, go to the `/src/` folder and clone this repository
3. In the root of the workspace, build everything as:
```
colcon build --symlink-install
```
5. Create a conda environment with simpy, scipy, and numpy

## Simulation Run and Communication (simulation)
1. Run the SITL simulation
```
make px4_sitl gazebo-classic
```
2. Run the Micro XRCE Agent as the bridge between ROS2 and PX4.
```
MicroXRCEAgent udp4 -p 8888
```


## Using The NR Controller Computation and Offboard Publisher
1. The length of time the algorithm runs before the land sequence begins may be changed via the variable in the **\_\_init\_\_** function at the top:   **self.time_before_land**
2. The reference path may be changed through the **self.main_traj** variable
3. The mass for the quadrotor may be changed for your specific hardware on the **elif not self.sim** statement in the **\_\_init\_\_** at the top. Don't change for simulation unless you change the simulation model explicitly.
4. The thrust/throttle mapping may be changed for your specific hardware on the **get_throttle_command_from_force** and **get_force_from_throttle_command** functions. Don't change for simulation unless you change the simulation model explicitly.

### Running the controller:
1. In another terminal tab, source the environment from the root of your ROS2 workspace: 
```
source install/setup.bash
```
2. Activate your conda environment in this same terminal with sourcing
3. Run the controller with the following command
```
ros2 run quad_newton_raphson_flow nr_quad log1.lo
```

# Blimp Code Instructions
## Running the Blimp Simulator
1. In a terminal tab, source the environment from the root of your ROS2 workspace: 
```
source install/setup.bash
```
2. Activate your conda environment in this same terminal with sourcing
3. In *run_blimp_sim.py* there is a dictionary named **mapping** with names of controller/path pairs. Let the desired pair be called **ctrl_path_name** and run the simulation by: 
```
ros2 run blimp_mpc_fbl_nr run_blimp_sim hardware_<\**ctrl_path_name**\>_circle_horz log1.log
```
## Citing this Work:
Please see arxiv preprint [here](https://arxiv.org/abs/2508.14185)  

## Authors:
Evanns G. Morales-Cuadrado, Luke Baird, Yorai Wardi, Samuel Coogan
