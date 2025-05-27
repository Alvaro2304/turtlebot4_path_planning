#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <memory>

class EKFNode : public rclcpp::Node
{
public:
    EKFNode() : Node("ekf_localization_node")
    {
        // Initialize EKF state [x, y, theta, vx, vy, vtheta]
        state_ = Eigen::VectorXd::Zero(6);
        
        // Initialize covariance matrix (6x6)
        covariance_ = Eigen::MatrixXd::Identity(6, 6) * 0.1;
        
        // Process noise covariance Q
        Q_ = Eigen::MatrixXd::Identity(6, 6);
        Q_(0, 0) = 0.01; // x position noise
        Q_(1, 1) = 0.01; // y position noise  
        Q_(2, 2) = 0.01; // theta noise
        Q_(3, 3) = 0.1;  // vx noise
        Q_(4, 4) = 0.1;  // vy noise
        Q_(5, 5) = 0.1;  // vtheta noise
        
        // Measurement noise covariance for odometry R_odom
        R_odom_ = Eigen::MatrixXd::Identity(6, 6);
        R_odom_(0, 0) = 0.05; // x measurement noise
        R_odom_(1, 1) = 0.05; // y measurement noise
        R_odom_(2, 2) = 0.02; // theta measurement noise
        R_odom_(3, 3) = 0.2;  // vx measurement noise
        R_odom_(4, 4) = 0.2;  // vy measurement noise
        R_odom_(5, 5) = 0.1;  // vtheta measurement noise
        
        // Measurement noise covariance for IMU R_imu (only angular velocity)
        R_imu_ = Eigen::MatrixXd::Identity(1, 1);
        R_imu_(0, 0) = 0.01; // angular velocity noise
        
        // TF Setup
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        // Subscribers
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&EKFNode::odomCallback, this, std::placeholders::_1));
            
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 10, std::bind(&EKFNode::imuCallback, this, std::placeholders::_1));
        
        // Publisher for fused odometry
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", 10);
            
        // Timer for EKF prediction (higher frequency)
        prediction_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), // 50 Hz
            std::bind(&EKFNode::predictionStep, this));
            
        last_prediction_time_ = this->get_clock()->now();
        
        RCLCPP_INFO(this->get_logger(), "EKF Localization Node Started");
    }

private:
    // EKF State: [x, y, theta, vx, vy, vtheta]
    Eigen::VectorXd state_;
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd Q_; // Process noise
    Eigen::MatrixXd R_odom_; // Odometry measurement noise
    Eigen::MatrixXd R_imu_;  // IMU measurement noise
    
    // TF components
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // ROS components
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr prediction_timer_;
    
    // Timing
    rclcpp::Time last_prediction_time_;
    bool first_odom_received_ = false;
    bool first_imu_received_ = false;
    
    // Latest sensor data
    nav_msgs::msg::Odometry::SharedPtr latest_odom_;
    sensor_msgs::msg::Imu::SharedPtr latest_imu_;
    
    // Motion model (constant velocity model)
    void motionModel(const Eigen::VectorXd& state, double dt, Eigen::VectorXd& predicted_state)
    {
        predicted_state = state;
        
        // Update position based on velocity
        predicted_state[0] += state[3] * cos(state[2]) * dt - state[4] * sin(state[2]) * dt; // x
        predicted_state[1] += state[3] * sin(state[2]) * dt + state[4] * cos(state[2]) * dt; // y
        predicted_state[2] += state[5] * dt; // theta
        
        // Normalize theta to [-pi, pi]
        while (predicted_state[2] > M_PI) predicted_state[2] -= 2.0 * M_PI;
        while (predicted_state[2] < -M_PI) predicted_state[2] += 2.0 * M_PI;
        
        // Velocities remain constant (could add acceleration model here)
        // predicted_state[3], predicted_state[4], predicted_state[5] unchanged
    }
    
    // Jacobian of the motion model
    Eigen::MatrixXd getMotionJacobian(const Eigen::VectorXd& state, double dt)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        
        double theta = state[2];
        double vx = state[3];
        double vy = state[4];
        
        // Partial derivatives
        F(0, 2) = -vx * sin(theta) * dt - vy * cos(theta) * dt; // dx/dtheta
        F(0, 3) = cos(theta) * dt;  // dx/dvx
        F(0, 4) = -sin(theta) * dt; // dx/dvy
        
        F(1, 2) = vx * cos(theta) * dt - vy * sin(theta) * dt;  // dy/dtheta
        F(1, 3) = sin(theta) * dt;  // dy/dvx
        F(1, 4) = cos(theta) * dt;  // dy/dvy
        
        F(2, 5) = dt; // dtheta/dvtheta
        
        return F;
    }
    
    // EKF Prediction Step
    void predictionStep()
    {
        if (!first_odom_received_) return;
        
        auto current_time = this->get_clock()->now();
        double dt = (current_time - last_prediction_time_).seconds();
        
        if (dt <= 0.0) return;
        
        // Predict state
        Eigen::VectorXd predicted_state(6);
        motionModel(state_, dt, predicted_state);
        
        // Predict covariance
        Eigen::MatrixXd F = getMotionJacobian(state_, dt);
        Eigen::MatrixXd predicted_covariance = F * covariance_ * F.transpose() + Q_ * dt;
        
        // Update
        state_ = predicted_state;
        covariance_ = predicted_covariance;
        
        last_prediction_time_ = current_time;
        
        // Publish transform and odometry
        publishResults(current_time);
    }
    
    // Odometry callback - full state update
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        latest_odom_ = msg;
        
        if (!first_odom_received_) {
            // Initialize state with first odometry measurement
            initializeState(msg);
            first_odom_received_ = true;
            return;
        }
        
        // Extract measurements [x, y, theta, vx, vy, vtheta]
        Eigen::VectorXd z_odom(6);
        z_odom[0] = msg->pose.pose.position.x;
        z_odom[1] = msg->pose.pose.position.y;
        
        // Convert quaternion to yaw
        tf2::Quaternion quat;
        tf2::fromMsg(msg->pose.pose.orientation, quat);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        z_odom[2] = yaw;
        
        z_odom[3] = msg->twist.twist.linear.x;
        z_odom[4] = msg->twist.twist.linear.y;
        z_odom[5] = msg->twist.twist.angular.z;
        
        // Measurement model (direct observation)
        Eigen::MatrixXd H_odom = Eigen::MatrixXd::Identity(6, 6);
        
        // EKF Update
        ekfUpdate(z_odom, H_odom, R_odom_);
        
        RCLCPP_DEBUG(this->get_logger(), "Odometry update: x=%.3f, y=%.3f, theta=%.3f", 
                     state_[0], state_[1], state_[2]);
    }
    
    // IMU callback - angular velocity update only
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        latest_imu_ = msg;
        
        if (!first_odom_received_) return; // Wait for odometry initialization
        
        first_imu_received_ = true;
        
        // Extract angular velocity measurement
        Eigen::VectorXd z_imu(1);
        z_imu[0] = msg->angular_velocity.z; // Assuming IMU is aligned with robot
        
        // Measurement model (observe angular velocity)
        Eigen::MatrixXd H_imu(1, 6);
        H_imu.setZero();
        H_imu(0, 5) = 1.0; // Observe vtheta
        
        // EKF Update
        ekfUpdate(z_imu, H_imu, R_imu_);
        
        RCLCPP_DEBUG(this->get_logger(), "IMU update: angular_vel=%.3f", z_imu[0]);
    }
    
    // Generic EKF update step
    void ekfUpdate(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
    {
        // Innovation (measurement residual)
        Eigen::VectorXd y = z - H * state_;
        
        // Handle angle wrapping for theta measurements
        if (H.rows() >= 3 && H(2, 2) == 1.0) { // If measuring theta directly
            while (y[2] > M_PI) y[2] -= 2.0 * M_PI;
            while (y[2] < -M_PI) y[2] += 2.0 * M_PI;
        }
        
        // Innovation covariance
        Eigen::MatrixXd S = H * covariance_ * H.transpose() + R;
        
        // Kalman gain
        Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();
        
        // Update state and covariance
        state_ = state_ + K * y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
        covariance_ = (I - K * H) * covariance_;
        
        // Normalize theta
        while (state_[2] > M_PI) state_[2] -= 2.0 * M_PI;
        while (state_[2] < -M_PI) state_[2] += 2.0 * M_PI;
    }
    
    // Initialize EKF state with first odometry measurement
    void initializeState(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        state_[0] = msg->pose.pose.position.x;
        state_[1] = msg->pose.pose.position.y;
        
        tf2::Quaternion quat;
        tf2::fromMsg(msg->pose.pose.orientation, quat);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        state_[2] = yaw;
        
        state_[3] = msg->twist.twist.linear.x;
        state_[4] = msg->twist.twist.linear.y;
        state_[5] = msg->twist.twist.angular.z;
        
        RCLCPP_INFO(this->get_logger(), "EKF initialized: x=%.3f, y=%.3f, theta=%.3f", 
                    state_[0], state_[1], state_[2]);
    }
    
    // Publish TF transform and odometry message
    void publishResults(const rclcpp::Time& timestamp)
    {
        // Publish TF transform
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = timestamp;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";
        
        transform.transform.translation.x = state_[0];
        transform.transform.translation.y = state_[1];
        transform.transform.translation.z = 0.0;
        
        tf2::Quaternion quat;
        quat.setRPY(0, 0, state_[2]);
        transform.transform.rotation = tf2::toMsg(quat);
        
        tf_broadcaster_->sendTransform(transform);
        
        // Publish filtered odometry
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = timestamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";
        
        // Pose
        odom_msg.pose.pose.position.x = state_[0];
        odom_msg.pose.pose.position.y = state_[1];
        odom_msg.pose.pose.position.z = 0.0;
        odom_msg.pose.pose.orientation = tf2::toMsg(quat);
        
        // Twist
        odom_msg.twist.twist.linear.x = state_[3];
        odom_msg.twist.twist.linear.y = state_[4];
        odom_msg.twist.twist.angular.z = state_[5];
        
        // Covariance (simplified - map 6x6 to 6x6 pose covariance)
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                odom_msg.pose.covariance[i * 6 + j] = covariance_(i, j);
                odom_msg.twist.covariance[i * 6 + j] = covariance_(i, j);
            }
        }
        
        odom_pub_->publish(odom_msg);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}