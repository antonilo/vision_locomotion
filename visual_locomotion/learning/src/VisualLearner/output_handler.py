import numpy as np
from visualoco_msgs.msg import VisuaLatent
import rospy
from scipy.special import softmax



class OutputHandler():
    def __init__(self, config):
        self.config = config
        self.initiate_publishers()

    def initiate_publishers(self):
        self.publish_net_output = False
        self.prob_fall_pub = rospy.Publisher(
                "/agile_locomotion/vis_pred_prob",
                VisuaLatent, queue_size=1, tcp_nodelay=True)

    def process_net_output(self, net_output):
        msg = VisuaLatent()
        visual_latent = np.zeros_like(net_output)
        visual_latent[:8] = net_output[:8]
        gamma = net_output[9:]
        visual_latent[8:-1] = gamma
        visual_latent[-1] = net_output[8]
        msg.vis_latent = visual_latent
        return msg

    def publish_output(self, msg):
        self.prob_fall_pub.publish(msg)
