#!/usr/bin/env python3

import argparse
import sys
import threading

import rospy
from std_msgs.msg import Empty
from VisualLearner.visual_predictor import VisualPredictor

sys.path.append("../")
from config.config import read_config


class Tester():
    def __init__(self, config):
        rospy.init_node('fall_pred_node', anonymous=True)
        rospy.Subscriber("agile_locomotion/walk", Empty,
                         self.start_pipeline, queue_size=1)
        self.config = config

    def start_pipeline(self, data):
        print("Starting to walk")
        self.predictor.start_walk()

    def run_test(self):
        self.predictor = VisualPredictor(self.config, mode="deploy")
        #rospy.signal_shutdown("Experiment Finished")


def main():
    parser = argparse.ArgumentParser(description='Test Racing network.')
    parser.add_argument('--config_file',
                        help='Path to config yaml', required=True)

    args = parser.parse_args()
    config_filepath = args.config_file
    config = read_config(config_filepath, mode='test')
    tester = Tester(config)
    tester_thread = threading.Thread(target=tester.run_test)
    tester_thread.start()

    def shutdown_hook():
        print("shutdown time!")
        try:
            tester_thread.join(timeout=1.0)
        except:
            pass

    rospy.on_shutdown(shutdown_hook)
    rospy.spin()
    print("main thread exited!")


if __name__ == "__main__":
    main()
