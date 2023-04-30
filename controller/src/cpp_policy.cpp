#include "agile_locomotion/cpp_policy.h"


Model::Model(const std::string& base_path, const std::string& policy_id, const std::string& policy_name, const double hip_j_angle, int inp_dim) {
 std::string fname_mean =  base_path + "/mean" + policy_id + ".csv";
 std::string fname_var = base_path + "/var" + policy_id + ".csv";
 std::string fname_policy = base_path + "/" + policy_name + "_" + policy_id + ".pt";

 std::cout << fname_policy << std::endl;
 std::cout << fname_mean << std::endl;
 std::cout << fname_var << std::endl;
 hip_joint_angle = hip_j_angle;

 try {
   // Deserialize the ScriptModule from a file using torch::jit::load().
   module = torch::jit::load(fname_policy);
   std::cout << "Model loaded from " << fname_policy << std::endl;
 }
 catch (const c10::Error& e) {
   std::cerr << "error loading the model\n";
 }

 // loading mean and variance
 read_one_line_csv(fname_mean, obs_mean);
 read_one_line_csv(fname_var, obs_var);
 if (inp_dim > 0){
     obs_dim = inp_dim;
     obs_mean.resize(obs_dim);
     obs_var.resize(obs_dim);
 }
 inputs_i = torch::ones({1, obs_dim});
}

// evaluation function. Takes observation --> normalizes --> makes prediction --> corrects for action mean and std
// check that false are as expected
std::vector<float> Model::evaluate(const std::vector<double>& inp_obs, bool log_action){
  std::vector<double> action_std{hip_joint_angle, hip_joint_angle, hip_joint_angle, hip_joint_angle,
                                 hip_joint_angle, hip_joint_angle, hip_joint_angle, hip_joint_angle, hip_joint_angle,
                                 hip_joint_angle, hip_joint_angle, hip_joint_angle};
  //std::cout << "Inp obs is " << inp_obs << std::endl;

  std::vector<torch::jit::IValue> inputs;
  std::vector<double> norm_obs = normalize_obs(inp_obs);
  copy_tensor(norm_obs, inputs_i);
  inputs.push_back(inputs_i);

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::vector<float> output_v(output[0].data_ptr<float>(), output[0].data_ptr<float>() + output[0].numel());

  // perform the logging and normalization only if the action policy is being called
	if(log_action){
        // logging the last action taken by the policy
	    last_action = output_v;
	    // normalizing the output of the policy
        for(int j=0; j < 12; j++){
            output_v[j] = output_v[j] * action_std[j] + action_mean[j];
        }
    }
    return output_v;
}

void Model::read_one_line_csv(const std::string& fname, std::vector<double> &outvar)
{
  std::fstream newfile;
  newfile.open(fname,std::ios::in);
  std::string fline;
  getline(newfile, fline);
  newfile.close();
  std::stringstream first_line(fline);
  std::string tp;
  while(getline(first_line, tp, ' ')){
      outvar.push_back(std::stod(tp));
  }
  std::cout << "read " << outvar.size() << " elements" <<std::endl;
  std::cout << outvar <<std::endl;
  obs_dim = outvar.size();
}

std::vector<double> Model::normalize_obs(const std::vector<double>& obs_to_norm)
{
  std::vector<double> norm_obs;
  for(int j = 0; j < obs_to_norm.size(); j++)
  {
      double norm_val_i = (obs_to_norm[j] - obs_mean[j]) / (sqrt(obs_var[j] + 1e-8));
      norm_val_i = std::max(-10., std::min(norm_val_i, 10.));
      norm_obs.push_back(norm_val_i);
  }
  return norm_obs;
}

void Model::copy_tensor(const std::vector<double> &vec, torch::Tensor &tnsr)
{
    for (int j=0; j < vec.size(); j++)
        tnsr[0][j] = (float)vec[j];
}


void Model::softmax(std::vector<float> &inp)
{
	float m, sum, constant;

	m = -INFINITY;
	for (auto it = begin(inp); it != end (inp); ++it) {
		if (m < *it) {
			m = *it;
		}
	}

	sum = 0.0;
	for (auto it = begin(inp); it != end (inp); ++it) {
		sum += exp(*it - m);
	}

	constant = m + log(sum);
	for (auto it = begin(inp); it != end (inp); ++it) {
	  *it = exp(*it - constant);
	}

}

