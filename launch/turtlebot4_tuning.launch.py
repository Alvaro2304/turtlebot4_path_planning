from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='true',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='maze_mod',
                          description='Ignition World'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
]

# Add pose arguments
for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():
    # Get package directories
    pkg_turtlebot4_path_planning = get_package_share_directory('turtlebot4_path_planning')


    # Launch file paths
    turtlebot4_ignition_launch = PathJoinSubstitution(
        [pkg_turtlebot4_path_planning, 'launch', 'turtlebot4_ignition.launch.py'])
    

    # Include TurtleBot4 Ignition launch
    turtlebot4_ignition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([turtlebot4_ignition_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('rviz', LaunchConfiguration('rviz')),
            ('world', LaunchConfiguration('world')),
            ('model', LaunchConfiguration('model')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw'))
        ]
    )

    # EKF Node
    ekf_node = Node(
        package='turtlebot4_path_planning',
        executable='ekf_imu_encoder',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )


    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(turtlebot4_ignition)
    ld.add_action(ekf_node)
    
    return ld