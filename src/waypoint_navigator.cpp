#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <yaml-cpp/yaml.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <lifecycle_msgs/srv/get_state.hpp>

using namespace std::chrono_literals;

class WaypointNavigator : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  WaypointNavigator()
  : Node("waypoint_navigator"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    this->declare_parameter<std::string>("waypoints_file", "");
    client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");
    nav2_state_client_ = this->create_client<lifecycle_msgs::srv::GetState>("/bt_navigator/get_state");

    amcl_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "amcl_pose", 10, std::bind(&WaypointNavigator::amclCallback, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(2s, std::bind(&WaypointNavigator::onTimer, this));
  }

private:
  rclcpp_action::Client<NavigateToPose>::SharedPtr client_;
  rclcpp::Client<lifecycle_msgs::srv::GetState>::SharedPtr nav2_state_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::vector<geometry_msgs::msg::PoseStamped> waypoints_;
  size_t current_wp_ = 0;

  bool initial_pose_received_ = false;
  bool sent_ = false;

  void amclCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr)
  {
    if (!initial_pose_received_) {
      initial_pose_received_ = true;
      RCLCPP_INFO(this->get_logger(), "Initial pose received from AMCL.");
    }
  }

  void onTimer()
  {
    //RCLCPP_INFO(this->get_logger(), "Timer callback triggered");

    if (!initial_pose_received_) {
      RCLCPP_WARN(this->get_logger(), "Waiting for initial pose from AMCL...");
      return;
    }

    if (!client_->action_server_is_ready()) {
      RCLCPP_WARN(this->get_logger(), "Waiting for action server...");
      return;
    }

    if (!tf_buffer_.canTransform("map", "base_link", tf2::TimePointZero, tf2::durationFromSec(1.0))) {
      RCLCPP_WARN(this->get_logger(), "Waiting for transform map -> base_link...");
      return;
    }

    if (!nav2_state_client_->service_is_ready()) {
      RCLCPP_WARN(this->get_logger(), "Waiting for bt_navigator/get_state service...");
      return;
    }

    auto request = std::make_shared<lifecycle_msgs::srv::GetState::Request>();
    nav2_state_client_->async_send_request(request,
      [this](rclcpp::Client<lifecycle_msgs::srv::GetState>::SharedFuture future)
      {
        try {
          if (future.get()->current_state.label != "active") {
            RCLCPP_WARN(this->get_logger(), "Nav2 is not active yet...");
            return;
          }

          if (!sent_) {
            if (!loadWaypointsFromFile()) {
              RCLCPP_ERROR(this->get_logger(), "Failed to load waypoints.");
              return;
            }
            sent_ = true;
            sendNextGoal();
          }

        } catch (const std::exception & e) {
          RCLCPP_ERROR(this->get_logger(), "Error checking nav2 state: %s", e.what());
        }
      });
  }

  bool loadWaypointsFromFile()
  {
    std::string file_path;
    this->get_parameter("waypoints_file", file_path);
    if (file_path.empty()) {
      RCLCPP_ERROR(this->get_logger(), "waypoints_file parameter is empty");
      return false;
    }

    try {
      YAML::Node config = YAML::LoadFile(file_path);
      auto yaml_wps = config["waypoints"];
      if (!yaml_wps || !yaml_wps.IsSequence()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid or missing 'waypoints' in YAML");
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

      RCLCPP_INFO(this->get_logger(), "Loaded %lu waypoints.", waypoints_.size());
      return true;

    } catch (const YAML::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "YAML Exception: %s", e.what());
      return false;
    }
  }

  void sendNextGoal()
  {
    if (current_wp_ >= waypoints_.size()) {
      RCLCPP_INFO(this->get_logger(), "All waypoints reached.");
      return;
    }

    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose = waypoints_[current_wp_];

    RCLCPP_INFO(this->get_logger(), "Sending goal %lu", current_wp_ + 1);

    auto options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    options.result_callback = std::bind(&WaypointNavigator::resultCallback, this, std::placeholders::_1);

    client_->async_send_goal(goal_msg, options);
  }

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
