#pragma once

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include <glog/logging.h>
#include <future>

#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Float32.h"
#include "geometry_msgs/Vector3.h"
#include "comm.h"
#include "agile_locomotion/low_level.h"
#include "agile_locomotion/cpp_policy.h"
#include "unitree_joystick.h"
#include "utils/logging.h"
#include "visualoco_msgs/Proprioception.h"
#include "visualoco_msgs/VisuaLatent.h"

namespace agile_locomotion {

class AgileLocomotion {
 public:
  AgileLocomotion(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  AgileLocomotion() : AgileLocomotion(ros::NodeHandle(), ros::NodeHandle("~")) {};

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Subscribers
  ros::Subscriber toggle_experiment_sub_;
  ros::Subscriber des_vel_sub_;
  ros::Subscriber des_lat_sub_;
  ros::Subscriber current_vel_sub_;

  // Publishers
  ros::Publisher prop_pub_;

  void startExecutionCallback(const std_msgs::EmptyConstPtr& msg);
  void VelocityCallback(const geometry_msgs::Vector3& msg);
  void networkCallback(const visualoco_msgs::VisuaLatentConstPtr& msg);

  void updateState();

  int startWalking();

  bool loadParameters();

  std::vector<double> concat_vec(const int& nvals, const bool use_vel);

  void set_command_lin_speed(const double& speed);
  void set_command_ang_speed(const double& speed);

  void moveAllPosition(const std::vector<float>& targetPos, double duration);

  // Logging Helpers
  std::shared_ptr<logging::Logging> logging_helper_;
  ros::Time time_start_logging_;



  std::deque< std::vector<double> > obs_vec;
  // initialize latent to nominal value
  //std::vector<float> geom_prop_latent = {0., 0.};
  //std::vector<float> latent = {0.,0.,0.,0.,0.,0.,0.,0., 0., 0., 0., 0., 0, 0., 0., 0., 0.}; // combination of prop and geom
  std::vector<float> latent; // combination of prop and geom
  std::vector<float> visual_latent; // contains everything predicted from vision. Varying size.
  std::vector<float> prop_latent;
  float abs_sum = 0;
  bool joypad_terminate = false;
  UNITREE_LEGGED_SDK::LowState recvd_state = {0};
  std::vector<float> action_scaling{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}; //12dim
  std::vector<double> rpy_recvd{0,0,0};
  std::vector<double> jt_angles{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}; //12dim
  std::vector<double> jt_vel{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}; 
  std::vector<double> command_onehot{-0.5, 0, -0.5, 0};
  std::vector<double> vel{0, 0, 0};

  // Keyboard
  xRockerBtnDataStruct _keyData;
  ros::Time time_last_joy;
  ros::Timer joystick_loop_timer_, state_sleeper_;
  void joyLoop(const ros::TimerEvent& time);

  // Constants
  static constexpr double joypad_timeout_ = 0.5;
  static constexpr double joypad_axes_zero_tolerance_ = 0.2;
  static constexpr double kLoopFrequency_ = 50.0;
  static constexpr double kVelocityCommandZeroThreshold_ = 0.03;
  
  // LowLevel Controller
  std::shared_ptr<LowLevel> lowlevel;
  //LowLevel lowlevel = LowLevel(UNITREE_LEGGED_SDK::LOWLEVEL, false);

  // Torch Models
  std::shared_ptr<Model> base_policy; // the controller
  std::shared_ptr<Model> vision_policy; // the controller
  std::shared_ptr<Model> adaptation_policy; // the z estimator for prop
  //std::shared_ptr<Model> geom_pred_policy; // the z estimator for geom
  std::shared_ptr<Model> predictor_policy; // fall estimator

  // Fall predictors
  std::future<std::vector<float>> future_predictor_t;
  std::future_status status_predictor;
  std::vector<float> fall_prob = {1.,0.};
  ros::Time time_last_vis_latent= ros::Time::now();

  /////////////////////////////////////
  // Parameters
  /////////////////////////////////////

  int latent_size = 11;
  int visual_latent_size;
  int n_g1;
  int single_geom_latent_size = 2; 
  int obDim = 42;

  float Kpval, Kdval, force_threshold, hip_joint_angle;  
  double lin_speed, ang_speed, max_lin_speed, max_ang_speed, conf_max_lin_speed;
  int tsteps, init_steps;
  std::string base_path, policy_id, vision_path;
  bool testing;
  bool person_tracking = false;
  bool using_vision;
  bool deploy_vision = false;
  std::vector<float> last_action = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  // Log related
  bool log = false;
  bool log_images_ = false;
  bool log_prob_fall = false;
  bool log_ready = false;
  std::string data_dir_;
  std::string curr_data_dir_;

};

}  // namespace agile_autonomy
