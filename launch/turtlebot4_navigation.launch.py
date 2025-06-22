from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    pkg_turtlebot4_path_planning  = get_package_share_directory('turtlebot4_path_planning')

    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_turtlebot4_path_planning, 'maps', 'maze_mod_map.yaml'),
        description='Full path to map yaml file'
    )

    wp_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value=os.path.join(pkg_turtlebot4_path_planning, 'config', 'waypoints.yaml'),
        description='Full path to waypoint yaml file'
    )

    nav2_arg=DeclareLaunchArgument(
        'nav2_params',
        default_value=os.path.join(pkg_turtlebot4_path_planning, 'config', 'nav2_params.yaml'),
        description='Path to Nav2 param file'
)

    return LaunchDescription([
        map_arg,
        wp_arg,
        nav2_arg,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'params_file': LaunchConfiguration('nav2_params'),
                'use_sim_time': 'true'
            }.items()
        ),

        Node(
            package='turtlebot4_path_planning',
            executable='waypoint_navigator',
            name='waypoint_navigator',
            output='screen',
            parameters=[{'waypoints_file': LaunchConfiguration('waypoints_file')},
                        {'use_sim_time': True}]
        ),
        

        Node(
            package='turtlebot4_path_planning',
            executable='initial_pose_publisher',
            name='initial_pose_publisher',
            output='screen',
            parameters=[{'waypoints_file': LaunchConfiguration('waypoints_file')},
                        {'use_sim_time': True}]
        )


        
    ])