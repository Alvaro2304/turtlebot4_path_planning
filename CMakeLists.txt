cmake_minimum_required(VERSION 3.8)
project(turtlebot4_path_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclpy REQUIRED)
find_package(urdf REQUIRED)
find_package(irobot_create_description REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ros_ign_interfaces REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(lifecycle_msgs REQUIRED)

add_executable(ekf_imu_encoder src/ekf_imu_encoder.cpp)
ament_target_dependencies(ekf_imu_encoder
  rclcpp tf2 tf2_ros tf2_geometry_msgs
  geometry_msgs nav_msgs sensor_msgs
)

add_executable(waypoint_navigator src/waypoint_navigator.cpp)
ament_target_dependencies(waypoint_navigator
  rclcpp rclcpp_action nav2_msgs geometry_msgs yaml-cpp tf2 tf2_ros tf2_geometry_msgs
  lifecycle_msgs
)

add_executable(initial_pose_publisher src/initial_pose_publisher.cpp)
ament_target_dependencies(initial_pose_publisher
  rclcpp
  geometry_msgs
  yaml-cpp
)

target_link_libraries(ekf_imu_encoder
  ${rclcpp_LIBRARIES}
  Eigen3::Eigen
)
target_link_libraries(waypoint_navigator yaml-cpp)

target_link_libraries(initial_pose_publisher yaml-cpp)


install(TARGETS
  ekf_imu_encoder
  waypoint_navigator
  initial_pose_publisher
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  meshes
  urdf
  worlds
  config
  maps
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
