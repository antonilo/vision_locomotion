#pragma once

#include <fstream>
#include <iomanip>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

namespace logging {
class Logging {
 public:
  Logging(const ros::NodeHandle& nh, const ros::NodeHandle& pnh,
          const bool log_images, const int z_dim, const int gamma_dim,
          const int n_g1);
  void createDirectories(const std::string data_dir,
                         std::string* curr_data_dir);
  void newLog(const std::string& filename);
  void closeLog();
  bool logState(const ros::Duration& log_time,
                const std::vector<double>& rpy_recvd,
                const std::vector<double>& joint_angles,
                const std::vector<double>& joint_vel,
                const std::vector<float>& last_action,
                const std::vector<double>& foot_force,
                const std::vector<double>& high_lev_command,
                const std::vector<float>& prop_latent,
                const std::vector<float>& control_latent,
                const std::vector<double>& torques,
                const std::vector<double>& imu,
                const float fall_prob);

 private:
  struct StreamWithFilename {
    std::ofstream filestream;
    std::string filename;
  };
  StreamWithFilename log_file_states_;
  void writeStateHeader();
  void imgCallback(const sensor_msgs::ImageConstPtr& msg);
  void depthCallback(const sensor_msgs::ImageConstPtr& msg);
  void frontimgCallback(const sensor_msgs::ImageConstPtr& msg);

  // Members
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::Subscriber img_sub_, depth_sub_, front_img_sub_;
  std::string img_dir_;
  bool log_images_;
  bool save_img_to_disk_ = false;
  int frame_counter_ = -1;
  int front_frame_counter_ = -1;
  int depth_frame_counter_ = -1;

  // Parameters
  int joint_dim =12;
  int action_dim=12;
  int force_dim=4;
  int command_dim=4;
  int imu_dim=6;
  int z_dim_;
  int gamma_dim_;
  int n_g1_;
};

}  // namespace logging
