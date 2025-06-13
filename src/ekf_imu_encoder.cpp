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
    EKFNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("ekf_localization_node",options)
    {
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        
        // Initialize EKF state [x, y, theta, v, omega]
        state_ = Eigen::VectorXd::Zero(5);
        
        // Initialize covariance matrix (5x5)
        covariance_ = Eigen::MatrixXd::Identity(5, 5) * 0.1;
        
        // Process noise covariance Q
        Q_ = Eigen::MatrixXd::Identity(5, 5);
        Q_(0, 0) = 0.00001; // x position noise
        Q_(1, 1) = 0.00001; // y position noise  
        Q_(2, 2) = 2.; // theta noise
        Q_(3, 3) = 0.8;  // v (linear velocity) noise
        Q_(4, 4) = 2.;  // omega (angular velocity) noise
        
        // Measurement noise covariance for odometry R_odom
        // Odometry gives us [x, y, theta, v, omega]
        R_odom_ = Eigen::MatrixXd::Identity(2, 2);
        //R_odom_(0, 0) = 10.; // x measurement noise
        //R_odom_(1, 1) = 10.; // y measurement noise
        //R_odom_(2, 2) = 10.; // theta measurement noise
        R_odom_(0, 0) = 0.5;  // v measurement noise
        R_odom_(1, 1) = 10.; // omega measurement noise
        
        // Measurement noise covariance for IMU
        R_imu_ = Eigen::MatrixXd::Identity(3, 3);
        R_imu_(0, 0) = 0.00001; // orientation noise
        R_imu_(1, 1) = 1.; // v noise (accel-derived)
        R_imu_(2,2) = 0.00001; // omega noise
        
        // TF Setup
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        // Replace your current subscriptions with:
        auto qos = rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);

        // Subscribers
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", qos, std::bind(&EKFNode::odomCallback, this, std::placeholders::_1));
            
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", qos, std::bind(&EKFNode::imuCallback, this, std::placeholders::_1));
        
        // Publisher for fused odometry
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", 10);
            
        // Timer for EKF prediction (higher frequency)
        prediction_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), // 50 Hz
            std::bind(&EKFNode::predictionStep, this));
            
        
        
        RCLCPP_INFO(this->get_logger(), "EKF Localization Node Started");
    }

private:
    // EKF State: [x, y, theta, v, omega]
    Eigen::VectorXd state_;
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd Q_; // Process noise
    Eigen::MatrixXd R_odom_; // Odometry measurement noise
    Eigen::MatrixXd R_imu_; //IMU measurement noise
    
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
    rclcpp::Time last_imu_accel_time_;
    bool first_odom_received_ = false;
    bool first_imu_received_ = false;
    
    // Latest sensor data
    nav_msgs::msg::Odometry::SharedPtr latest_odom_;
    sensor_msgs::msg::Imu::SharedPtr latest_imu_;
    
    // Motion model (constant velocity model)
    void motionModel(const Eigen::VectorXd& state, double dt, Eigen::VectorXd& predicted_state)
    {
        predicted_state = state;
        
        double x = state[0];
        double y = state[1]; 
        double theta = state[2];
        double v = state[3];      // linear velocity
        double omega = state[4];  // angular velocity
        
        // Differential drive kinematics
        predicted_state[0] = x + v * cos(theta) * dt;  // x position
        predicted_state[1] = y + v * sin(theta) * dt;  // y position  
        predicted_state[2] = theta + omega * dt;       // orientation
        
        // Normalize theta to [-pi, pi]
        while (predicted_state[2] > M_PI) predicted_state[2] -= 2.0 * M_PI;
        while (predicted_state[2] < -M_PI) predicted_state[2] += 2.0 * M_PI;
        
        // Velocities remain constant
        //predicted_state[3] = v;     // unchanged
        //predicted_state[4] = omega; // unchanged
    }
    
    // Jacobian of the motion model
    Eigen::MatrixXd getMotionJacobian(const Eigen::VectorXd& state, double dt)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(5, 5);
        
        double theta = state[2];
        double v = state[3];
        
        // Partial derivatives for differential drive kinematics
        F(0, 2) = -v * sin(theta) * dt; // dx/dtheta
        F(0, 3) = cos(theta) * dt;      // dx/dv
        
        F(1, 2) = v * cos(theta) * dt;  // dy/dtheta  
        F(1, 3) = sin(theta) * dt;      // dy/dv
        
        F(2, 4) = dt;                   // dtheta/domega
        
        return F;
    }
    
    // EKF Prediction Step
    void predictionStep()
    {
        if (!first_odom_received_){
            //RCLCPP_INFO(this->get_logger(), "Prediction step skipped - no odom yet");
            return;
        }
        
        //RCLCPP_INFO(this->get_logger(), "Running prediction step");

        auto current_time = this->get_clock()->now();

        // Initialize last_prediction_time_ if this is the first prediction after initialization
        if (last_prediction_time_.seconds() == 0.0) {
            last_prediction_time_ = current_time;
            //RCLCPP_INFO(this->get_logger(), "Initializing prediction timer");
            return;
        }

        double dt = (current_time - last_prediction_time_).seconds();
        
        if (dt <= 1e-6) {  // Use small epsilon instead of 0.0
        RCLCPP_DEBUG(this->get_logger(), "Skipping prediction: dt too small (%.9f)", dt);
        return;  // Don't update last_prediction_time_
        }
        
        if (dt > 1.0) {
            RCLCPP_WARN(this->get_logger(), "Large dt detected: %.6f, resetting timer", dt);
            last_prediction_time_ = current_time;
            return;
        }
        
        // Add minimum dt threshold for stability
        if (dt < 0.001) {  // Less than 1ms
            RCLCPP_DEBUG(this->get_logger(), "Small dt: %.6f, skipping prediction", dt);
            return;
        }

        
        // Predict state
        Eigen::VectorXd predicted_state(5);
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
        //RCLCPP_INFO(this->get_logger(), "Received odometry message");

        latest_odom_ = msg;
        
        if (!first_odom_received_) {
            // Initialize state with first odometry measurement
            initializeState(msg);
            first_odom_received_ = true;
            RCLCPP_INFO(this->get_logger(), "First odom received and initialized"); 
            return;
        }
        
        // Extract measurements [x, y, theta, vx, omega]
        Eigen::VectorXd z_odom(2);
        //z_odom[0] = msg->pose.pose.position.x;
        //z_odom[1] = msg->pose.pose.position.y;
        
        // Convert quaternion to yaw
        //tf2::Quaternion quat;
        //tf2::fromMsg(msg->pose.pose.orientation, quat);
        //double roll, pitch, yaw;
        //tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        //z_odom[2] = yaw;
        
        z_odom[0] = msg->twist.twist.linear.x;
        z_odom[1] = msg->twist.twist.angular.z;
        
        // Measurement model (observe angular velocity)
        Eigen::MatrixXd H_odom(2, 5);
        H_odom.setZero();
        H_odom(0, 3) = 1.0; // Observe v
        H_odom(1, 4) = 1.0; // Observe omega

        
        // EKF Update
        ekfUpdate(z_odom, H_odom, R_odom_);
        
        //RCLCPP_DEBUG(this->get_logger(), "Odometry update: x=%.3f, y=%.3f, theta=%.3f", 
                     //state_[0], state_[1], state_[2]);
    }
    
    // IMU callback - angular velocity update only
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        //RCLCPP_INFO(this->get_logger(), "Received IMU message");
        latest_imu_ = msg;
        
        if (!first_odom_received_) return; // Wait for odometry initialization

        if (!first_imu_received_) {
            last_imu_accel_time_ = rclcpp::Time(msg->header.stamp);
            first_imu_received_ = true;
            return;
        }
        
        rclcpp::Time current_time = rclcpp::Time(msg->header.stamp);
        double dt = (current_time - last_imu_accel_time_).seconds();
        
        // Sanity checks
        if (dt <= 0.0 || dt > 0.1) {
            last_imu_accel_time_ = current_time;
            return;
        }

        double ax_robot = msg->linear_acceleration.x; // Forward acceleration
        
        // Extract orientation measurement
        Eigen::VectorXd z_imu(3);
        tf2::Quaternion quat;
        tf2::fromMsg(msg->orientation, quat);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        z_imu[0] = yaw;
        
        double predicted_v_change = ax_robot * dt;
        double expected_new_v = state_[3] + predicted_v_change;
        z_imu[1] = expected_new_v;

        z_imu[2] = msg->angular_velocity.z;

        
        
        // Measurement model (observe angular velocity)
        Eigen::MatrixXd H_imu(3, 5);
        H_imu.setZero();
        H_imu(0, 2) = 1.0; // Observe theta
        H_imu(1, 3) = 1.0; // Observe velocity
        H_imu(2, 4) = 1.0; // Observe omega
        
        // EKF Update
        ekfUpdate(z_imu, H_imu, R_imu_);
        
        RCLCPP_DEBUG(this->get_logger(), "IMU update: angular_vel=%.3f", z_imu[2]);
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
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5, 5);
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
        state_[4] = msg->twist.twist.angular.z;
        
        RCLCPP_INFO(this->get_logger(), "EKF initialized: x=%.3f, y=%.3f, theta=%.3f", 
                    state_[0], state_[1], state_[2]);
    }
    
    // Publish TF transform and odometry message
    void publishResults(const rclcpp::Time& timestamp)
    {
        // Use the latest sensor timestamp instead of prediction time when available
        rclcpp::Time publish_time = timestamp;

        // If we have recent sensor data, use its timestamp
        //if (latest_odom_ && (timestamp - latest_odom_->header.stamp).seconds() < 0.1) {
        //publish_time = latest_odom_->header.stamp;
    //}
        
        // Publish TF transform
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = publish_time;
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
        odom_msg.header.stamp = publish_time;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";
        
        // Pose
        odom_msg.pose.pose.position.x = state_[0];
        odom_msg.pose.pose.position.y = state_[1];
        odom_msg.pose.pose.position.z = 0.0;
        odom_msg.pose.pose.orientation = tf2::toMsg(quat);
        
        // Twist
        odom_msg.twist.twist.linear.x = state_[3];
        odom_msg.twist.twist.linear.y = 0.0;
        odom_msg.twist.twist.angular.z = state_[4];
        
        // Covariance (map 5x5 to 6x6 pose/twist covariance matrices)
        // Initialize covariance arrays to zero
        std::fill(odom_msg.pose.covariance.begin(), odom_msg.pose.covariance.end(), 0.0);
        std::fill(odom_msg.twist.covariance.begin(), odom_msg.twist.covariance.end(), 0.0);
        
        // Map relevant covariances
        odom_msg.pose.covariance[0] = covariance_(0, 0);   // x-x
        odom_msg.pose.covariance[7] = covariance_(1, 1);   // y-y  
        odom_msg.pose.covariance[35] = covariance_(2, 2);  // yaw-yaw
        
        odom_msg.twist.covariance[0] = covariance_(3, 3);  // vx-vx
        odom_msg.twist.covariance[35] = covariance_(4, 4); // vyaw-vyaw
        
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
