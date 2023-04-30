#pragma once

#include <torch/script.h> // One-stop header.
#include <glog/logging.h>
#include <vector>
#include <memory>
#include <deque>
#include <string>
#include <stdio.h>


class Model{
 public:
  Model(const std::string& base_path, const std::string& policy_id,
        const std::string& policy_name, const double hip_j_angle, int inp_dim = -1);
  std::vector<float> operator () (const std::vector<double>& inp_obs, bool log_action = false) {
    return evaluate(inp_obs, log_action);
  }

  // Parameters
  torch::jit::script::Module module;
  std::vector<double> obs_mean;
  std::vector<double> obs_var;
  torch::Tensor inputs_i;
  std::vector<float> action_mean = {0.05, 0.8, -1.4, -0.05, 0.8, -1.4, 0.05, 0.8, -1.4, -0.05, 0.8, -1.4};
  int obs_dim;
  std::vector<float> last_action = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  std::vector<float> evaluate(const std::vector<double>& inp_obs, bool log_action = false);
  void softmax(std::vector<float>& inp);

 private:
  void read_one_line_csv(const std::string& fname, std::vector<double>& outvar);
  std::vector<double> normalize_obs(const std::vector<double>& obs_to_norm);
  void copy_tensor(const std::vector<double> &vec, torch::Tensor &tnsr);
  double hip_joint_angle;
};
