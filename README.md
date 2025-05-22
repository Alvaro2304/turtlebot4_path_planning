# Turtlebot4 Navigation Sandbox

A ROS2 package containing various navigation nodes for TurtleBot4, including localization and different path planning methods for experimentation and development.

## Requirements

- **ROS2 Humble**
- **Gazebo Ignition Fortress**

## Prerequisites

Before using this package, you need to install the official TurtleBot4 packages. Follow the installation instructions from the official TurtleBot4 documentation:

🔗 [TurtleBot4 Installation Guide](https://turtlebot.github.io/turtlebot4-user-manual/software/overview.html)

## Package Contents

This package will provide:
- **Localization nodes** for TurtleBot4 positioning
- **Multiple path planning algorithms** for navigation testing


## Installation

1. Clone this repository into your ROS2 workspace:
```bash
cd ~/your_ros2_ws/src
git clone https://github.com/Alvaro2304/turtlebot4_path_planning.git
```

2. Build the package:
```bash
cd ~/your_ros2_ws
colcon build --packages-select turtlebot4_path_planning
```

3. Source your workspace:
```bash
source ~/your_ros2_ws/install/setup.bash
```

## Usage

*I will update this part as soon I'll develop the nodes*

## Known Issues

### Gazebo Simulation Issues

**RPLidar not publishing data:**
- **Problem**: The TurtleBot4's RPLidar sensor may not publish any data in Gazebo simulation
- **Solution**: Ensure that Ignition Gazebo is running with GPU acceleration enabled, as the LiDAR plugin uses GPU-based ray casting for simulation
- **How to verify**: Check that your system has proper GPU drivers installed and Ignition is utilizing GPU resources

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TurtleBot4 development team
- ROS2 community
- Open Robotics
