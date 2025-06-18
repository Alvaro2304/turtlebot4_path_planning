
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>
#include <lifecycle_msgs/srv/get_state.hpp>

using namespace std::chrono_literals;

class WaypointNavigator : public rclcpp::Node
{
public:
  // Aliases for action types
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  WaypointNavigator()
  : Node("waypoint_navigator"),
    tf_buffer_(this->get_clock()),  // Initialize TF buffer with the node's clock
    tf_listener_(tf_buffer_)        // Attach a transform listener to the buffer
  {
    // Declare a ROS parameter for the YAML file containing waypoints
    this->declare_parameter<std::string>("waypoints_file", "");

    // Set up client to communicate with Nav2's "navigate_to_pose" action server
    client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

    // Periodic timer to attempt goal sending logic every 2 seconds
    timer_ = this->create_wall_timer(
      2s, std::bind(&WaypointNavigator::timerCallback, this));

    // Subscribe to AMCL pose topic to detect when initial pose has been set
    amcl_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "amcl_pose", 10,
      [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr)
      {
        if (!initial_pose_received_) {
          initial_pose_received_ = true;
          RCLCPP_INFO(this->get_logger(), "Initial pose received from AMCL.");
        }
      });
  }

private:
  // Client to send navigation goals
  rclcpp_action::Client<NavigateToPose>::SharedPtr client_;
  // Periodic timer
  rclcpp::TimerBase::SharedPtr timer_;
  // List of waypoints to visit
  std::vector<geometry_msgs::msg::PoseStamped> waypoints_;
  // Index of the current waypoint
  size_t current_wp_ = 0;

  // TF buffer/listener to handle map-to-base transforms
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Subscription to detect when AMCL has received initial pose
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_;

  // State flags
  bool sent_ = false;
  bool initial_pose_received_ = false;

  // === Helper: Check if bt_navigator is active via lifecycle service ===
  bool isNav2Active()
  {
    auto client = this->create_client<lifecycle_msgs::srv::GetState>("/bt_navigator/get_state");
    if (!client->wait_for_service(1s)) {
      RCLCPP_WARN(this->get_logger(), "Timed out waiting for bt_navigator/get_state service.");
      return false;
    }

    auto request = std::make_shared<lifecycle_msgs::srv::GetState::Request>();
    auto future = client->async_send_request(request);

    // Spin until we get a response or timeout
    if (rclcpp::spin_until_future_complete(shared_from_this(), future, 2s)
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_WARN(this->get_logger(), "Failed to call get_state on bt_navigator.");
      return false;
    }

    auto state = future.get()->current_state.label;
    return state == "active";
  }

  // === Timer callback: Drives waypoint sending logic ===
  void timerCallback()
  {
    //RCLCPP_INFO(this->get_logger(), "Timer callback triggered");

    // Wait for action server (navigate_to_pose) to be available
    if (!client_->wait_for_action_server(1s)) {
      RCLCPP_WARN(this->get_logger(), "Waiting for navigate_to_pose action server...");
      return;
    }

    // Wait until Nav2 stack is active (esp. bt_navigator)
    if (!isNav2Active()) {
      RCLCPP_WARN(this->get_logger(), "Nav2 not active. Waiting...");
      return;
    }

    // Ensure AMCL has a pose (user has clicked "2D Pose Estimate")
    if (!initial_pose_received_) {
      RCLCPP_WARN(this->get_logger(), "Waiting for initial pose from AMCL...");
      return;
    }

    // Ensure we can transform between map and base_link (TF is live)
    if (!tf_buffer_.canTransform("map", "base_link", tf2::TimePointZero, tf2::durationFromSec(1.0))) {
      RCLCPP_WARN(this->get_logger(), "Waiting for map -> base_link transform...");
      return;
    }

    // If not already sent, load waypoints and start navigation
    if (!sent_) {
      if (!loadWaypointsFromFile()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load waypoints from file.");
        return;
      }

      sent_ = true;
      sendNextGoal();
    }
  }

  // === Load waypoints from YAML file ===
  bool loadWaypointsFromFile()
  {
    std::string file_path;
    this->get_parameter("waypoints_file", file_path);
    if (file_path.empty()) {
      RCLCPP_ERROR(this->get_logger(), "waypoints_file parameter is empty.");
      return false;
    }

    try {
      YAML::Node config = YAML::LoadFile(file_path);
      auto yaml_wps = config["waypoints"];

      if (!yaml_wps || !yaml_wps.IsSequence()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid or missing 'waypoints' in YAML.");
        return false;
      }

      for (const auto & wp : yaml_wps) {
        if (wp.size() != 3) continue;
        double x = wp[0].as<double>();
        double y = wp[1].as<double>();
        double theta = wp[2].as<double>();

        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.orientation.w = cos(theta / 2.0);
        pose.pose.orientation.z = sin(theta / 2.0);

        waypoints_.push_back(pose);
      }

      RCLCPP_INFO(this->get_logger(), "Loaded %lu waypoints from file.", waypoints_.size());
      return true;

    } catch (const YAML::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "YAML parsing error: %s", e.what());
      return false;
    }
  }

  // === Send the next navigation goal ===
  void sendNextGoal()
  {
    if (current_wp_ >= waypoints_.size()) {
      RCLCPP_INFO(this->get_logger(), "All waypoints reached.");
      return;
    }

    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose = waypoints_[current_wp_];

    RCLCPP_INFO(this->get_logger(), "Sending goal %lu", current_wp_ + 1);

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.result_callback =
      std::bind(&WaypointNavigator::resultCallback, this, std::placeholders::_1);

    client_->async_send_goal(goal_msg, send_goal_options);
  }

  // === Callback to handle result of navigation ===
  void resultCallback(const GoalHandleNavigateToPose::WrappedResult & result)
  {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_INFO(this->get_logger(), "Reached waypoint %lu", current_wp_ + 1);
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to reach waypoint %lu", current_wp_ + 1);
    }

    current_wp_++;
    sendNextGoal();
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WaypointNavigator>());
  rclcpp::shutdown();
  return 0;
}
