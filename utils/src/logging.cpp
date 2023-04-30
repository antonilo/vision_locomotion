#include "utils/logging.h"
#include <algorithm>
#include <experimental/filesystem>
#include <boost/filesystem.hpp>

namespace logging {

Logging::Logging(const ros::NodeHandle& nh, const ros::NodeHandle& pnh,
                 const bool log_images, const int z_dim,
                 const int gamma_dim, const int n_g1)
                 : nh_(nh), pnh_(pnh), log_images_(log_images) {

 z_dim_ = z_dim; 
 gamma_dim_ = gamma_dim;
 n_g1_ = n_g1;
 
 // ROS publishers and subscribers
 if (log_images_) {
   image_transport::ImageTransport it(pnh_);
   img_sub_ = it.subscribe(
    "/d400/color/image_raw", 1, &Logging::imgCallback, this);
   depth_sub_ = it.subscribe(
    "/d400/depth/image_rect_raw", 1, &Logging::depthCallback, this);
   front_img_sub_ = it.subscribe(
    "/front/color/image_raw", 1, &Logging::frontimgCallback, this);
 }
} 

void Logging::createDirectories(const std::string data_dir,
                                std::string* curr_data_dir) {
  std::printf("Creating directories in [%s]\n", data_dir.c_str());
  // create directory
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%y-%m-%d_%H-%M-%S");
  auto time_str = oss.str();
  std::string rollout_dir = data_dir + "/rollout_" + time_str;

  boost::filesystem::create_directory(rollout_dir);
  if (log_images_) {
    img_dir_ = rollout_dir + "/img";
    boost::filesystem::create_directory(rollout_dir + "/img");
  }

  *curr_data_dir = rollout_dir;
}

void Logging::newLog(const std::string& filename) {
  log_file_states_.filename = filename;
  log_file_states_.filestream.open(log_file_states_.filename,
                                     std::ios_base::app | std::ios_base::in);
  writeStateHeader();
  printf("Opened new states file: %s\n", log_file_states_.filename.c_str());
  // time to log images
  if (log_images_) save_img_to_disk_ = true;
}

void Logging::closeLog() {
  printf("Close State Log File.\n");
  if (log_file_states_.filestream.is_open()) {
    log_file_states_.filestream.close();
  }
  save_img_to_disk_ = false;
}

void Logging::writeStateHeader() {
  // clang-format off
  log_file_states_.filestream << "time_from_start" << ","
                                << "rpy_0" << ","
                                << "rpy_1" << ",";
  for (size_t i = 0; i < joint_dim; i++)
  {
   log_file_states_.filestream << "joint_angles_" << i << ","; 
  }
  for (size_t i = 0; i < joint_dim; i++)
  {
   log_file_states_.filestream << "joint_vel_" << i << ","; 
  }
  for (size_t i = 0; i < joint_dim; i++)
  {
   log_file_states_.filestream << "torque_" << i << ","; 
  }
  for (size_t i = 0; i < action_dim; i++)
  {
   log_file_states_.filestream << "last_action_" << i << ","; 
  }
  for (size_t i = 0; i < force_dim; i++)
  {
   log_file_states_.filestream << "foot_force_" << i << ","; 
  }
  for (size_t i = 0; i < command_dim; i++)
  {
   log_file_states_.filestream << "command_" << i << ","; 
  }
  for (size_t i = 0; i < (z_dim_); i++)
  {
   log_file_states_.filestream << "prop_latent_" << i << ","; 
  }
  for (size_t i = 0; i < (z_dim_ + n_g1_ - 1); i++)
  {
   log_file_states_.filestream << "control_latent_" << i << ","; 
  }
  for (size_t i = 0; i < (3); i++)
  {
   log_file_states_.filestream << "imu_g_" << i << ","; 
  }
  for (size_t i = 0; i < (3); i++)
  {
   log_file_states_.filestream << "imu_a_" << i << ","; 
  }
  log_file_states_.filestream << "fall_prob" << ","; 
  log_file_states_.filestream << "frame_counter" << ","; 
  log_file_states_.filestream << "depth_frame_counter" << ","; 
  log_file_states_.filestream << "front_frame_counter" << "\n"; 
  // clang-format on
}

bool Logging::logState(const ros::Duration& log_time,
                       const std::vector<double>& rpy_recvd,
                       const std::vector<double>& joint_angles,
                       const std::vector<double>& joint_vel,
                       const std::vector<float>& last_action,
                       const std::vector<double>& foot_force,
                       const std::vector<double>& high_lev_command,
                       const std::vector<float>& prop_latent,
                       const std::vector<float>& vision_latent,
                       const std::vector<double>& torques,
                       const std::vector<double>& imu,
                       const float fall_prob) {
  if (log_file_states_.filestream.is_open()) {
    // clang-format off
    log_file_states_.filestream << std::fixed
                                << std::setprecision(8) << log_time.toSec() << ","
                                << std::setprecision(8) << rpy_recvd[0] << ","
                                << std::setprecision(8) << rpy_recvd[1] << ",";
    for (size_t i = 0; i < joint_dim; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << joint_angles[i] << ","; 
    }
    for (size_t i = 0; i < joint_dim; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << joint_vel[i] << ","; 
    }
    for (size_t i = 0; i < joint_dim; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << torques[i] << ","; 
    }
    for (size_t i = 0; i < action_dim; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << last_action[i] << ","; 
    }
    for (size_t i = 0; i < force_dim; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << foot_force[i] << ","; 
    }
    for (size_t i = 0; i < command_dim ; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << high_lev_command[i] << ","; 
    }
    for (size_t i = 0; i < (z_dim_) ; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << prop_latent[i] << ","; 
    }
    for (size_t i = 0; i < (z_dim_ + n_g1_ -1) ; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << vision_latent[i] << ","; 
    }
    for (size_t i = 0; i < (imu_dim) ; i++)
    {
     log_file_states_.filestream << std::setprecision(8) << imu[i] << ","; 
    }
    log_file_states_.filestream << std::setprecision(8) << fall_prob << ","; 
    log_file_states_.filestream << std::setprecision(8) << frame_counter_ << ","; 
    log_file_states_.filestream << std::setprecision(8) << depth_frame_counter_ << ","; 
    log_file_states_.filestream << std::setprecision(8) << front_frame_counter_ << "\n"; 
    log_file_states_.filestream.flush();
    // clang-format on
    return true;
  }
  return false;
}

void Logging::imgCallback(const sensor_msgs::ImageConstPtr& msg) {
  if (!save_img_to_disk_) return;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  frame_counter_ += 1;
  std::ostringstream ss;
  ss << std::setw(8) << std::setfill('0') << frame_counter_;
  std::string s2(ss.str());

  // save image to disk
  std::string rgb_img_filename =
      img_dir_ + "/frame_" + s2 + ".jpg";

  cv::imwrite(rgb_img_filename, cv_ptr->image);
}

void Logging::frontimgCallback(const sensor_msgs::ImageConstPtr& msg) {
  if (!save_img_to_disk_) return;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  front_frame_counter_ += 1;
  std::ostringstream ss;
  ss << std::setw(8) << std::setfill('0') << front_frame_counter_;
  std::string s2(ss.str());

  // save image to disk
  std::string rgb_img_filename =
      img_dir_ + "/front_frame_" + s2 + ".jpg";

  cv::imwrite(rgb_img_filename, cv_ptr->image);
}

void Logging::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
  if (!save_img_to_disk_) return;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  depth_frame_counter_ += 1;
  std::ostringstream ss;
  ss << std::setw(8) << std::setfill('0') << depth_frame_counter_;
  std::string s2(ss.str());

  // save image to disk
  std::string depth_img_filename =
      img_dir_ + "/depth_" + s2 + ".tiff";

  cv::imwrite(depth_img_filename, cv_ptr->image);
}
}  // namespace logging
