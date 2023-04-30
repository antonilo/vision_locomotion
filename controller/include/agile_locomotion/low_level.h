#include "agile_locomotion/unitree_legged_sdk.h"
#include "comm.h"
#include <math.h>

class LowLevel {
  public:
    LowLevel(uint8_t level, bool isTestFlag);
    void UDPRecv(){udp.Recv();}
    void UDPSend(){udp.Send();}
    void publish_cmd(const std::vector<float>& joints,
                     const float pgain,
                     const float dgain);
    void getState(UNITREE_LEGGED_SDK::LowState& state, std::vector<double> &rpy);
    void quatToEuler(const std::vector<double> &quat, std::vector<double> &eulerVec);

    // Parameters
  private:
    bool isTest = false;
    int runcounter = 0;
    UNITREE_LEGGED_SDK::Safety safe;
    UNITREE_LEGGED_SDK::UDP udp;
    UNITREE_LEGGED_SDK::LowCmd cmd = {0};
    UNITREE_LEGGED_SDK::LowState state = {0};

};
