#include "agile_locomotion/low_level.h"
#include <iostream>


LowLevel::LowLevel(uint8_t level, bool isTestFlag):
          safe(UNITREE_LEGGED_SDK::LeggedType::A1), udp(level){
  udp.InitCmdData(cmd);
	isTest = isTestFlag;
}


void LowLevel::publish_cmd(const std::vector<float>& joints, const float pgain,
                           const float dgain){

    std::vector<float> pdTau(12,0);
    std::vector<float> Kp_vector = {pgain, pgain, pgain};
    std::vector<float> Kd_vector = {dgain, dgain, dgain};
    //std::cout << "command, ";
    for(int j=0; j < 12; j++)
    {
       float action_i = joints[j];
       pdTau[j] = (50 * (action_i - state.motorState[j].q) + 0.2 * (-state.motorState[j].dq));
       cmd.motorCmd[j].q = action_i;//2.146E+9f;
       cmd.motorCmd[j].dq = 0;
       cmd.motorCmd[j].Kp = Kp_vector[j%3];
       cmd.motorCmd[j].Kd = Kd_vector[j%3];
       cmd.motorCmd[j].tau = 0;
       //std::cout << action_i << " ";
    }
    //std::cout << "Pd: " << pdTau << std::endl;
    safe.PositionLimit(cmd);
    safe.PowerProtect(cmd, state, 9);
    if (!isTest){
      udp.SetSend(cmd);
    }
      runcounter += 1;
};

void LowLevel::quatToEuler(const std::vector<double> &quat, std::vector<double> &eulerVec) {
    double qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    // roll (x-axis rotation)
    double sinr_cosp = 2 * (qw * qx + qy * qz);
    double cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    eulerVec[0] = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1)
        eulerVec[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        eulerVec[1] = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (qw * qz + qx * qy);
    double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    eulerVec[2] = std::atan2(siny_cosp, cosy_cosp);
}


void LowLevel::getState(UNITREE_LEGGED_SDK::LowState& state, std::vector<double> &rpy){
    udp.GetRecv(state);
    std::vector<double> quat{state.imu.quaternion[0], state.imu.quaternion[1], state.imu.quaternion[2], state.imu.quaternion[3]};
    quatToEuler(quat, rpy);
    // Acceleration state.imu.accelerometer[0], state.imu.accelerometer[1], state.imu.accelerometer[2]
    // Motor Position rpy[:12]
    // Motor Velocity rpy[12:]
    // Quat quat{state.imu.quaternion[0], state.imu.quaternion[1], state.imu.quaternion[2], state.imu.quaternion[3]};
    // FootContact quat{state.imu.quaternion[0], state.imu.quaternion[1], state.imu.quaternion[2], state.imu.quaternion[3]};
}
