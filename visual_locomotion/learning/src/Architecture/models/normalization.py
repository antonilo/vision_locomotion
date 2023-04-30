import numpy as np


class Normalization:
    def __init__(self, normalization_coeffs):
        self.normalization_coeffs = [c.astype(np.float32) for c in normalization_coeffs]

    def get_coeffs(self):
        return self.normalization_coeffs

    def normalize_inputs(self, prop):
        '''

        :param features:
        :return:
        '''
        prop = (prop - self.normalization_coeffs[0]) / (self.normalization_coeffs[1] + 1e-10)
        return prop

    def normalize_labels(self, labels):
        '''

        :param labels:
        :return:
        '''
        labels = (labels - self.normalization_coeffs[2]) / self.normalization_coeffs[3]
        return labels

    def unnormalize_inputs(self, prop):
        '''
        :param features:
        :return:
        '''
        prop = prop * self.normalization_coeffs[1] + self.normalization_coeffs[0]
        return prop

    def unnormalize_labels(self, labels):
        '''

        :param labels:
        :return:
        '''
        labels = labels * self.normalization_coeffs[3] + self.normalization_coeffs[2]
        return labels
