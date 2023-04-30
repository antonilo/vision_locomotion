#include "agile_locomotion/agile_locomotion.h"
#include "agile_locomotion/cpp_policy.h"
#include <vector>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <deque>
#include <fcntl.h>
#include <stdio.h>
#include <string>
#include <atomic>
#include <ctime>
#include <deque>
#include "quadrotor_common/parameter_helper.h"

using namespace UNITREE_LEGGED_SDK;
using boost::asio::ip::udp;
using namespace std::chrono;

namespace agile_locomotion
{


  AgileLocomotion::AgileLocomotion(const ros::NodeHandle &nh,
                                   const ros::NodeHandle &pnh)
      : nh_(nh), pnh_(pnh)
  {
    if (!loadParameters())
    {
      ROS_ERROR("[%s] Failed to load all parameters",
                ros::this_node::getName().c_str());
      ros::shutdown();
    }

    // Build some parameters (8 ext, 1 is slope below)
    visual_latent_size = latent_size + n_g1 - 1;

    for (int i = 0; i < visual_latent_size; i++)
    {
      visual_latent.push_back(0.0);
    }

    for (int i = 0; i < latent_size; i++)
    {
      latent.push_back(0.0);
      prop_latent.push_back(0.0);
    }

    // Subscriber
    toggle_experiment_sub_ = nh_.subscribe("agile_locomotion/walk", 1,
                                           &AgileLocomotion::startExecutionCallback, this);
    des_lat_sub_ = nh_.subscribe("agile_locomotion/vis_pred_prob", 1,
                                 &AgileLocomotion::networkCallback, this);

    // Publisher
    prop_pub_ = nh_.advertise<visualoco_msgs::Proprioception>(
        "agile_locomotion/proprioception", 1);
    ROS_INFO("Ros is Ready");

    // Logger
    logging_helper_ =
        std::make_shared<logging::Logging>(nh_, pnh_, log_images_,
                                           latent_size, single_geom_latent_size, n_g1);

    // Build low level controller and model
    lowlevel = std::make_shared<LowLevel>(UNITREE_LEGGED_SDK::LOWLEVEL, testing);

    base_policy = std::make_shared<Model>(base_path, policy_id, "mlp",
                                          hip_joint_angle, obDim + latent_size);
    if (using_vision)
    {
      vision_policy = std::make_shared<Model>(vision_path, policy_id, "mlp",
                                            hip_joint_angle, obDim + visual_latent_size);
    }
    adaptation_policy = std::make_shared<Model>(base_path, policy_id, "prop_encoder",
    	                                            hip_joint_angle, obDim * tsteps);
  }

  void AgileLocomotion::startExecutionCallback(const std_msgs::EmptyConstPtr &msg)
  {
    ROS_INFO("Received startExecutionCallback message!");
    // Add joystick stuff
    joystick_loop_timer_ = nh_.createTimer(ros::Duration(1.0 / kLoopFrequency_),
                                           &AgileLocomotion::joyLoop, this);
    startWalking();
    if (log)
      logging_helper_->closeLog();
  }

  void AgileLocomotion::networkCallback(const visualoco_msgs::VisuaLatentConstPtr &msg)
  {
    for (int i = 0; i < visual_latent_size; i++)
    {
      visual_latent[i] = msg->vis_latent[i];
    }
    time_last_vis_latent = ros::Time::now();
  }

  void AgileLocomotion::joyLoop(const ros::TimerEvent &time)
  {
    bool joypad_available = (ros::Time::now() - time_last_joy) <
                            ros::Duration(joypad_timeout_);
    if (joypad_available)
    {
      // control angular velocity
      if (_keyData.btn.components.B == 1)
      {
        joypad_terminate = true;
      }
      if (_keyData.btn.components.A == 1)
      {
        max_lin_speed = 0.32;
      }
      if (_keyData.btn.components.Y == 1)
      {
        max_lin_speed = 0.48;
      }
      if (_keyData.btn.components.X == 1)
      {
        max_lin_speed = 1.0;
      }
      // control angular velocity
      lin_speed = max_lin_speed * fabs((double)_keyData.ly);
	  lin_speed = std::max(0.05, lin_speed);
      ang_speed = -1. * max_ang_speed * (double)_keyData.lx;
      set_command_lin_speed(lin_speed);
      set_command_ang_speed(ang_speed);
    }
    else
    {
      // joypad not there, stopping
      set_command_ang_speed(0.0);
      set_command_lin_speed(0.05);
    }
  }

  std::vector<double> AgileLocomotion::concat_vec(const int &nvals, const bool use_vel)
  {
    int N = obs_vec.size();
    std::vector<double> hist_vec(obs_vec[N - nvals]);
    for (int i = 1; i < nvals; i++)
    {
      hist_vec.insert(hist_vec.end(), obs_vec[N - nvals + i].begin(), obs_vec[N - nvals + i].end());
    }

    if (use_vel)
      hist_vec.insert(hist_vec.end(), vel.begin(), vel.end());
    return hist_vec;
  }

  void AgileLocomotion::set_command_lin_speed(const double &speed)
  {
    // std::cout << "Lin speed is " <<speed << std::endl;
    double scaled_input = speed - 0.5;
    command_onehot[0] = scaled_input;
    command_onehot[2] = scaled_input;
  }

  void AgileLocomotion::set_command_ang_speed(const double &speed)
  {
    // std::cout << "Ang speed is " <<speed << std::endl;
    command_onehot[1] = speed;
    command_onehot[3] = speed;
  }

  void AgileLocomotion::updateState()
  {

    // Read state
    lowlevel->getState(recvd_state, rpy_recvd);

    // Use it to update things we need. TODO: write a class that handles all of it.
    std::vector<double> imu{recvd_state.imu.gyroscope[0],
						    recvd_state.imu.gyroscope[1],
                            recvd_state.imu.gyroscope[2],
                            recvd_state.imu.accelerometer[0],
                            recvd_state.imu.accelerometer[1],
                            recvd_state.imu.accelerometer[2]};


    std::vector<double> foot_force(4, 0);
    std::vector<double> foot_force_mag(4, 0);
    std::vector<double> foot_force_est(4, 0);
    for (int j = 0; j < 4; j++)
    {
      foot_force[j] = (double)(recvd_state.footForce[j] > force_threshold);
      foot_force_mag[j] = (double)(recvd_state.footForce[j]);
      foot_force_est[j] = (double)(recvd_state.footForceEst[j]);
    }
    std::vector<double> joint_angles(12, 0);
    std::vector<double> joint_vel(12, 0);
    std::vector<double> torques(12, 0);
    std::vector<double> quaternion(4, 0);
    std::vector<double> acceleration(3, 0);

    for (int j = 0; j < 3; j++)
    {
      acceleration[j] = recvd_state.imu.accelerometer[j];
    }

    for (int j = 0; j < 4; j++)
    {
      quaternion[j] = recvd_state.imu.quaternion[j];
    }

    abs_sum = 0;
    for (int j = 0; j < 12; j++)
    {
      joint_angles[j] = recvd_state.motorState[j].q;
      joint_vel[j] = recvd_state.motorState[j].dq;
      torques[j] = recvd_state.motorState[j].tauEst;
      abs_sum += abs(recvd_state.motorState[j].q);
    }

    // Copy at the end to avoid race conditions
    jt_angles = joint_angles;
    jt_vel = joint_vel;

    // joystick update loop
    memcpy(&_keyData, recvd_state.wirelessRemote, 40);
    if (fabs(_keyData.lx >
             joypad_axes_zero_tolerance_) ||
        fabs(_keyData.ly >
             joypad_axes_zero_tolerance_) ||
        _keyData.btn.components.B == 1)
    {
      time_last_joy = ros::Time::now();
    }

    std::vector<double> obs_i;
    std::copy(rpy_recvd.begin(), rpy_recvd.begin() + 2, std::back_inserter(obs_i));
    std::copy(joint_angles.begin(), joint_angles.end(), std::back_inserter(obs_i));
    std::copy(joint_vel.begin(), joint_vel.end(), std::back_inserter(obs_i));
    std::copy(last_action.begin(), last_action.end(), std::back_inserter(obs_i));
    std::copy(command_onehot.begin(), command_onehot.end(), std::back_inserter(obs_i));
    obs_vec.push_back(obs_i);

    // Log stuff
    if (log && log_ready)
    {
      ros::Duration log_time = ros::Time::now() - time_start_logging_;
      std::vector<float> full_prop_latent = prop_latent;
      bool success = logging_helper_->logState(log_time,
                                               rpy_recvd,
                                               joint_angles,
                                               joint_vel,
                                               last_action,
                                               foot_force,
                                               command_onehot,
                                               full_prop_latent,
                                               visual_latent,
                                               torques,
                                               imu,
                                               fall_prob[1]);
      if (!success)
        ROS_INFO("Logging Failed.");
    }

    // Publish proprioception and prop-latent
    visualoco_msgs::Proprioception prop_msg;
    prop_msg.header.stamp = ros::Time::now();
    prop_msg.rpy_0 = (float)rpy_recvd[0];
    prop_msg.rpy_1 = (float)rpy_recvd[1];
    prop_msg.fall_prob = (float)fall_prob[1];
    for (int j = 0; j < 12; j++)
    {
      prop_msg.joint_angles.push_back((float)joint_angles[j]);
      prop_msg.joint_vel.push_back((float)joint_vel[j]);
      prop_msg.last_action.push_back(last_action[j]);
    }
    for (int j = 0; j < 4; j++)
    {
      prop_msg.foot_force.push_back((float)foot_force[j]);
      prop_msg.command.push_back((float)command_onehot[j]);
      prop_msg.quaternion.push_back((float)quaternion[j]);
    }
    for (int j = 0; j < 3; j++)
    {
      prop_msg.acceleration.push_back((float)acceleration[j]);
    }
    for (int i = 0; i < latent_size; i++)
    {
      prop_msg.prop_latent.push_back(prop_latent[i]);
    }

    prop_pub_.publish(prop_msg);
  }

  int AgileLocomotion::startWalking()
  {
    // Set speeds. Later controlled by a ros node.
    set_command_lin_speed(lin_speed);
    set_command_ang_speed(ang_speed);

    // mean and var values for the latent
	if (using_vision) {
	 for (int i = obDim; i < obDim + visual_latent_size; i++)
     {
       vision_policy->obs_mean[i] = 0.0;
       vision_policy->obs_var[i] = 1.0;
	 }
	}
    for (int i = obDim; i < obDim + latent_size; i++)
    {
      	base_policy->obs_mean[i] = 0.0;
     	base_policy->obs_var[i] = 1.0;
    }

    InitEnvironment();
    LoopFunc loop_udpSend("udp_send", 0.002, 3, boost::bind(&LowLevel::UDPSend, lowlevel));
    LoopFunc loop_udpRecv("udp_recv", 0.002, 3, boost::bind(&LowLevel::UDPRecv, lowlevel));
    loop_udpSend.start();
    loop_udpRecv.start();

    std::vector<float> zero_action(12, 0);
    // TODO: put testing into parameters
    while (abs_sum == 0 && !testing)
    {
      lowlevel->publish_cmd(zero_action, 0.0, 0.0);
      updateState();
      usleep(10 * 1000);
    }

    usleep(2000 * 1000);
    updateState();
    moveAllPosition(base_policy->action_mean, 500);
    base_policy->last_action = base_policy->action_mean;
	if (using_vision)
      vision_policy->last_action = base_policy->action_mean;
    last_action = base_policy->last_action;
    for (int j = 0; j < 105; j++)
    {
      updateState();
      usleep(2 * 1000);
    }
    updateState();

    // Params for init stroll
    float base_pgain = 25;
    float base_dgain = 2;

    // Adaptation Prop
    bool use_vel = false;
    std::future<std::vector<float>> future_t1 = std::async(std::launch::async, *adaptation_policy, concat_vec(tsteps, use_vel));
    prop_latent = future_t1.get();
    for (int i = 0; i < latent_size; i++)
    {
      latent[i] = prop_latent[i];
    }
    future_t1 = std::async(std::launch::async, *adaptation_policy, concat_vec(tsteps, use_vel));
    std::future_status status;

    // Ready to Log
    if (log)
    {
      std::cout << "START_LOG_" << std::endl;
      logging_helper_->createDirectories(data_dir_, &curr_data_dir_);
      logging_helper_->newLog(curr_data_dir_ + "/proprioception.csv");
      time_start_logging_ = ros::Time::now();
      log_ready = true;
    }

    for (int run_i = 0;; run_i++)
    {
      auto start = high_resolution_clock::now();

      updateState();
      if (std::abs(rpy_recvd[0]) > 1.0 || std::abs(rpy_recvd[1]) > 1.0 || joypad_terminate)
        return 1;
      std::vector<double> obs_latent_vec(obs_vec[obs_vec.size() - 1]);
      bool vision_available = (ros::Time::now() - time_last_vis_latent) <
                                    ros::Duration(joypad_timeout_);
      std::vector<float> output_v;

	  if (vision_available) {
        obs_latent_vec.insert(obs_latent_vec.end(), visual_latent.begin(), visual_latent.end());
        output_v = vision_policy->evaluate(obs_latent_vec, true);
      	last_action = vision_policy->last_action;
		    //std::cout << "Using vision" << std::endl;
      } else {
		    //std::cout << "Walking blind " << std::endl;
        obs_latent_vec.insert(obs_latent_vec.end(), latent.begin(), latent.end());
        output_v = base_policy->evaluate(obs_latent_vec, true);
      	last_action = base_policy->last_action;
	  }


      if (run_i < init_steps)
      {
        lowlevel->publish_cmd(output_v, base_pgain, base_dgain);
        std::cout << "In init" << std::endl;
      }
      else
      {
        lowlevel->publish_cmd(output_v, Kpval, Kdval);
        status = future_t1.wait_for(std::chrono::milliseconds(1));
        if (status == std::future_status::ready)
        {
          prop_latent = future_t1.get();
          future_t1 = std::async(std::launch::async, *adaptation_policy, concat_vec(tsteps, use_vel));
          for (int i = 0; i < latent_size; i++)
          {
            latent[i] = prop_latent[i];
          }
        }
      }

      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      double exec_time = ((double)duration.count()) / 1000000.;
      // std::cout << "Exec time " << exec_time << std::endl;
      if (exec_time < 0.01)
      {
        std::this_thread::sleep_for(microseconds(10000 - duration.count()));
      }
      //auto control_stop = high_resolution_clock::now();
      //duration = duration_cast<microseconds>(control_stop - start);
      //std::cout << "one control cycle time, " << ((double)duration.count()) / 1000000. << std::endl;
    }
  }

  void AgileLocomotion::moveAllPosition(const std::vector<float> &targetPos, double duration)
  {
    float pgain = 50;
    float dgain = 1;
    float pos[12], lastPos[12], percent;
    for (int j = 0; j < 12; j++)
      lastPos[j] = recvd_state.motorState[j].q;
    std::vector<float> action_output(12, 0);
    for (int i = 1; i <= duration; i++)
    {
      updateState();
      percent = (float)i / duration;
      for (int j = 0; j < 12; j++)
      {
        action_output[j] = lastPos[j] * (1 - percent) + targetPos[j] * percent;
      }
      lowlevel->publish_cmd(action_output, pgain, dgain);
      usleep(5 * 1000);
    }
    for (int i = 0; i < 200; i++)
    {
      lowlevel->publish_cmd(action_output, pgain, dgain);
      usleep(5 * 1000);
    }
  }

  bool AgileLocomotion::loadParameters()
  {
    ROS_INFO("Loading parameters...");
    if (!quadrotor_common::getParam("base_path", base_path))
      return false;
    if (!quadrotor_common::getParam("vision_path", vision_path))
      return false;
    if (!quadrotor_common::getParam("general/policy_id", policy_id))
      return false;
    if (!quadrotor_common::getParam("general/Kp", Kpval))
      return false;
    if (!quadrotor_common::getParam("general/Kd", Kdval))
      return false;
    if (!quadrotor_common::getParam("general/force_threshold", force_threshold))
      return false;
    if (!quadrotor_common::getParam("general/hip_joint_angle", hip_joint_angle))
      return false;
    if (!quadrotor_common::getParam("general/lin_speed",
                                    lin_speed, 0.12))
      return false;
    if (!quadrotor_common::getParam("general/max_lin_speed",
                                    conf_max_lin_speed, 0.7))
      return false;
    max_lin_speed = conf_max_lin_speed;
    if (!quadrotor_common::getParam("general/ang_speed",
                                    ang_speed, 0.0))
      return false;
    if (!quadrotor_common::getParam("general/max_ang_speed",
                                    max_ang_speed, 0.5))
      return false;
    if (!quadrotor_common::getParam("general/tsteps",
                                    tsteps, 10))
      return false;
    if (!quadrotor_common::getParam("general/init_steps",
                                    init_steps, 10))
      return false;
    if (!quadrotor_common::getParam("general/testing",
                                    testing, false))
      return false;
    if (!quadrotor_common::getParam("general/log",
                                    log, false))
      return false;
    if (!quadrotor_common::getParam("general/latent_size",
                                    latent_size, 10))
      return false;
    if (!quadrotor_common::getParam("general/obDim",
                                    obDim, 42))
      return false;
    if (!quadrotor_common::getParam("visual_pred/n_g1",
                                    n_g1, 2))
      return false;
    if (!quadrotor_common::getParam("visual_pred/using_vision",
                                    using_vision, false))
      return false;
    if (log)
    {
      if (!quadrotor_common::getParam("general/log_images",
                                      log_images_, false))
        return false;
      // Strings are treated separately
      if (!pnh_.getParam("data_dir", data_dir_))
      {
        return false;
      }
    }

    ROS_INFO("Parameters Loaded");
    return true;
  }
} // namespace agile_locomotion

int main(int argc, char **argv)
{

  ros::init(argc, argv, "agile_locomotion");
  agile_locomotion::AgileLocomotion agile_loc;

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  return 0;
}
