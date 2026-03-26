# LRS Robotics Lab Setup — Tic-Tac-Toe Robot

This guide describes how to run the Tic-Tac-Toe Robot system in the **LRS Robotics Lab environment** using the Franka Emika Panda robot and RealSense camera.

---

## Prerequisites

Before starting, ensure:

* The robot is powered on and connected
* You are on the correct network
* ROS Noetic workspace is built:

  ```bash
  cd ~/ttt_ws
  catkin_make
  source devel/setup.bash
  ```
* RealSense camera is connected
* YOLO model is placed at:

  ```
  ttt_robot/models/best.pt
  ```

---

## System Startup (7 Terminals)

Open **7 separate terminals** and run the following commands **in order**:

---

### Terminal 1 — ROS Core

```bash
roscore
```

---

### Terminal 2 — Franka Control

```bash
roslaunch franka_control franka_control.launch robot_ip:=172.16.0.2
```

---

### Terminal 3 — Controller Spawner

```bash
rosrun controller_manager spawner position_joint_trajectory_controller
```

---

### Terminal 4 — MoveIt

```bash
roslaunch panda_moveit_config move_group.launch transmission:=position
```

---

### Terminal 5 — RealSense Camera

```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

---

### Terminal 6 — Robot Description

```bash
cd ~/ttt_ws
source devel/setup.bash
roslaunch ttt_robot ttt_robot_description.launch
```

---

### Terminal 7 — Main System (Game + Vision + Motion)

```bash
cd ~/ttt_ws
source devel/setup.bash
roslaunch ttt_robot ttt_7nodes.launch
```

---

## Game Execution

1. Wait until all nodes are initialized
2. The system enters idle state
3. Human makes a move
4. Press:

   ```
   N + Enter
   ```
5. Robot:

   * Moves to scan pose
   * Detects board state
   * Computes next move
   * Picks and places token

Repeat until the game ends.

---

## Important Notes

* Always start terminals in the correct order
* Do **not** skip controller spawning
* Ensure TF chain is valid:

  ```
  panda_link0 → panda_hand → camera_link → camera_color_optical_frame
  ```
* YOLO runs inside a virtual environment (if configured)
* If vision fails → check RealSense topics
* If motion fails → check MoveIt and controllers

---

## Known Issues

* RealSense may occasionally report:

  ```
  HW not ready
  ```

  → Restart camera launch

* Missing TF transforms:
  → Check static transform publisher

* PyKDL not available in venv:
  → Base-frame pose publishing may fail

---

## Shutdown

To safely stop the system:

1. Stop `ttt_7nodes.launch`
2. Stop MoveIt and controllers
3. Stop `franka_control`
4. Shutdown `roscore`

---

## Notes for Lab Users

This setup is **specific to the LRS Robotics Lab**:

* Robot IP: `172.16.0.2`
* Workspace: `~/ttt_ws`
* Pre-calibrated board and camera setup assumed

---

