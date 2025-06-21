#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

using namespace std::chrono_literals;

class InitialPosePublisher : public rclcpp::Node
{
public:
  InitialPosePublisher()
  : Node("initial_pose_publisher")
  {
    // Declare and get the YAML file parameter
    this->declare_parameter<std::string>("waypoints_file", "");
    std::string yaml_path = this->get_parameter("waypoints_file").as_string();

    // Load initial pose from YAML
    if (!loadInitialPoseFromFile(yaml_path)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load initial pose. Exiting.");
      rclcpp::shutdown();
      return;
    }

    // Publisher to /initialpose
    pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("initialpose", 10);

    // Publish after short delay
    timer_ = this->create_wall_timer(2s, std::bind(&InitialPosePublisher::publishInitialPose, this));
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg_;
  bool published_ = false;

  bool loadInitialPoseFromFile(const std::string & path)
  {
    try {
      YAML::Node config = YAML::LoadFile(path);
      if (!config["initial_pose"] || config["initial_pose"].size() != 3) {
        RCLCPP_ERROR(this->get_logger(), "Missing or malformed 'initial_pose' in YAML.");
        return false;
      }

      double x = config["initial_pose"][0].as<double>();
      double y = config["initial_pose"][1].as<double>();
      double theta = config["initial_pose"][2].as<double>();

      pose_msg_.header.frame_id = "map";
      pose_msg_.pose.pose.position.x = x;
      pose_msg_.pose.pose.position.y = y;
      pose_msg_.pose.pose.orientation.w = cos(theta / 2.0);
      pose_msg_.pose.pose.orientation.z = sin(theta / 2.0);

      // Optional: small covariance to avoid AMCL rejecting it
      pose_msg_.pose.covariance[0] = 0.25;
      pose_msg_.pose.covariance[7] = 0.25;
      pose_msg_.pose.covariance[35] = 0.06853891945200942;  // ~15 degrees

      return true;
    } catch (const YAML::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "YAML parsing error: %s", e.what());
      return false;
    }
  }

  void publishInitialPose()
  {
    if (published_) return;

    pose_msg_.header.stamp = this->now();
    pub_->publish(pose_msg_);
    RCLCPP_INFO(this->get_logger(), "Initial pose published.");
    published_ = true;
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<InitialPosePublisher>());
  rclcpp::shutdown();
  return 0;
}
