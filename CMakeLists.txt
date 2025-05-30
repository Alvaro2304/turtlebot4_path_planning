cmake_minimum_required(VERSION 3.8)
project(turtlebot4_path_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(urdf REQUIRED)
find_package(irobot_create_description REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ros_ign_interfaces REQUIRED)


add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(listener src/subscriber_member_function.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

add_executable(ekf_imu_encoder src/ekf_imu_encoder.cpp)
ament_target_dependencies(ekf_imu_encoder
  rclcpp tf2 tf2_ros tf2_geometry_msgs
  geometry_msgs nav_msgs sensor_msgs
)

target_link_libraries(ekf_imu_encoder
  ${rclcpp_LIBRARIES}
  Eigen3::Eigen
)

install(TARGETS
  talker
  listener
  ekf_imu_encoder
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  meshes
  urdf
  worlds
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
