# Tic-Tac-Toe Robot (ROS Noetic)

This project implements a fully autonomous Tic-Tac-Toe playing robotic system using a **Franka Emika Panda** robot, **ROS Noetic**, and **MoveIt**. The system integrates vision-based perception, game logic, and motion planning to interact with a physical game board.

---

## System Overview

The robot observes the board using a **RealSense D455 camera**, detects tokens using a **YOLO-based vision system**, determines the next move using a **minimax algorithm**, and executes pick-and-place actions using **MoveIt motion planning**.

---

## Main Components

### 1. Vision System

* **Node:** `vision_yolo_token_pick_node3.py`
* Detects tokens (X / O) using YOLO
* Computes:

  * 3D position (camera frame & base frame)
  * Token orientation (yaw)
* Publishes:

  * `/ttt/token_best_pose_cam`
  * `/ttt/token_best_pose_base`
  * `/ttt/token_best_yaw_deg`

---

### 2. Board Scanner

* **Node:** `ttt_board_scanner.py`
* Uses calibrated cell positions in the **camera frame**
* Detects board state (X / O / empty)
* Publishes:

  * `/ttt/board_state`

---

### 3. Game Manager

* **Node:** `ttt_game_manager.py`
* Implements game logic using **minimax**
* Manages state machine:

  * WAIT_HUMAN → SCAN → DECIDE → EXECUTE
* Publishes:

  * `/ttt/robot_move_cmd`
  * `/ttt/vision_target_token`

---

### 4. Motion Executor

* **Node:** `ttt_motion_executor.py`
* Controls robot movement using MoveIt
* Executes:

  * Token picking
  * Placement on board
* Uses:

  * Pre-calibrated board cell poses (base frame)
* Subscribes:

  * `/ttt/token_target_pose_base`
  * `/ttt/token_best_yaw_deg`

---

### 5. TF Transformation Node

* **Node:** `token_cam_to_base_tf3.py`
* Converts poses from camera frame → base frame

---

## Calibration

Calibration is performed using **AprilTags** placed in each board cell.

* Camera frame → used for perception
* Base frame → used for motion planning

Outputs:

* Cell positions in base frame (`cell_poses_base.json`)
* Accurate board plane estimation

---

## Project Structure

```
ttt_robot/
├── action/
├── config/
├── launch/
├── msg/
├── scripts/
├── urdf/
├── CMakeLists.txt
├── package.xml
├── README.md
└── .gitignore
```

---

## Requirements

* Ubuntu + ROS Noetic
* MoveIt
* `franka_ros`
* `panda_moveit_config`
* `realsense2_camera`
* Python 3
* Virtual environment for YOLO (Ultralytics)

---

## Dependencies

This project depends on both ROS packages and Python packages.

### Main ROS dependencies
- ROS Noetic
- MoveIt
- franka_ros
- panda_moveit_config
- realsense2_camera
- tf / tf2
- cv_bridge

### Main Python dependencies
- numpy
- opencv-python
- ultralytics
- torch
- torchvision
- torchaudio
- scikit-learn
- pyrealsense2
- matplotlib

See `package.xml` for ROS package dependencies and `requirements.txt` for Python dependencies.

## Setup

### 1. Clone repository into workspace

```bash
cd ~/ttt_ws/src
git clone https://github.com/ZhantlekSaduakas/ttt_robot.git
cd ..
catkin_make
source devel/setup.bash
```

---

### 2. Setup Python environment (YOLO)

```bash
cd ~/ttt_ws
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy
```

---

### 3. Place YOLO model

The model is **not included** in this repository.

Place it here:

```
ttt_robot/models/best.pt
```

---

## Running the System

### Start sequence:


Terminal 1
```bash
roscore
```
Terminal 2
```bash
roslaunch franka_control franka_control.launch robot_ip:=<ROBOT_IP>
```
Terminal 3
```bash
rosrun controller_manager spawner position_joint_trajectory_controller
```
Terminal 4
```bash
roslaunch panda_moveit_config move_group.launch transmission:=position
```
Terminal 5
```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```
Terminal 6
```bash
cd ~/ttt_ws
source devel/setup.bash
roslaunch ttt_robot ttt_robot_description.launch
```
Terminal 7
```bash
cd ~/ttt_ws
source devel/setup.bash
roslaunch ttt_robot ttt_7nodes.launch
```

---

## Game Flow

1. Human makes a move
2. Press **N + Enter**
3. Robot:

   * Moves to scan pose
   * Detects board state
   * Computes next move
   * Picks token
   * Places token
4. Repeat until game ends

---

## Notes

* YOLO runs inside a **virtual environment**
* Ensure TF chain is valid:

  ```
  panda_link0 → panda_hand → camera_link → camera_color_optical_frame
  ```
* Model file (`best.pt`) must be added manually
* Calibration must be performed before running

---

## Future Improvements

* Better grasping strategy
* Robust multi-object detection
* Improved error handling
* Full autonomy without keyboard input

---

## Author

Zhantlek Saduakas

Simin Bakhtiar

Robotics Project — Tic-Tac-Toe Robot (ROS Noetic)

---

## License

For academic and research purposes.
