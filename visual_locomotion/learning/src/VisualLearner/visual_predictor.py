import time
import rospy

from VisualLearner import input_handler
from VisualLearner import output_handler


class VisualPredictor():
    def __init__(self, config, mode='deploy'):
        self.config = config
        self.mode = mode
        self.time_start = rospy.Time()

        self.input_handler = input_handler.InputHandler(self.config)
        self.output_handler = output_handler.OutputHandler(self.config)


        self.initialize_timers()

    def initialize_timers(self):
        self.timer_pred = rospy.Timer(rospy.Duration(1. / 90),
                                      self.latent_prediction)

    def debug_print(self, string):
        debug = False
        if debug:
            print(string)

    def start_walk(self):
        self.time_start = rospy.get_rostime()
        self.output_handler.publish_net_output = True
        print("No problem in start")

    def latent_prediction(self, _event):
        if not self.output_handler.publish_net_output:
            return
        start = time.time()
        input_dict = self.input_handler.prepare_net_inputs()
        #print("Prediction time: Img timestamp - Now {}".format(self.input_handler.img_timestamp - rospy.Time.now().to_sec()))
        if (rospy.Time.now().to_sec() - self.input_handler.img_timestamp) > 100.0 and \
                self.config.input_use_imgs:
            print("Not received an image for some time. Not predicting anything")
            print("Now: {}; Img {}".format(rospy.Time.now().to_sec(), self.input_handler.img_timestamp))
            return
        prediction = self.input_handler.learner.inference(input_dict)
        pred_prob = self.output_handler.process_net_output(prediction)
        self.output_handler.publish_output(pred_prob)
        self.debug_print("Perception + Processing Latency is {}".format(rospy.Time.now().to_sec()-input_dict['img_ts']))
        self.debug_print("Processing pipeline has latency of {}".format(time.time() - start))
