from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms, utils


import os
from PIL import Image, ImageOps
import cv2
import random
import copy
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class VisionPropDataset(Dataset):

    def __init__(self, root_path, config, mode='train', debug=False):

        self.config = config
        self.mode = mode
        self.root_path = root_path
        self.normalization_coeffs = None
        self.debug = debug
        self.experiments = []
        self.last_frame_idx = 0
        self.input_prop_fts = []
        self.images_path = []
        self.label_features = []
        self.inputs = []
        self.targets = []
        self._add_features()
        self.rollout_boundaries = [0] # boundaries between different runs
        self.rollout_ts = []
        self.prop_latent = []
        self.setup_data_pipeline()
        if mode == 'deploy':
            return
        self.num_samples = 0
        file_rootname = 'rollout_'
        print("------------------------------------------")
        print('Building %s Dataset' % ("Training" if mode == 'train' else "Validation"))
        print("------------------------------------------")
        if os.path.isdir(root_path):
            for root, dirs, files in os.walk(root_path, topdown=True, followlinks=True):
                for name in dirs:
                    if name.startswith(file_rootname):
                        self.experiments.append(os.path.join(root, name))
        else:
            assert False, "Provided dataset root is neither a file nor a directory!"

        self.num_experiments = len(self.experiments)
        assert self.num_experiments > 0, 'No valid data found!'
        print('Dataset contains %d experiments.' % self.num_experiments)

        # numpy arrays to store the raw data and compute mean/std for normalization
        self.raw_prop_inputs = None

        print("Decoding data...")
        self.experiments = sorted(self.experiments)
        for exp in tqdm(self.experiments):
            try:
               self._decode_experiment(exp)
            except:
               print("Decoding failed in folder {}".format(exp))

        if len(self.inputs) == 0:
            raise IOError("Did not find any file in the dataset folder")
        self.inputs = torch.from_numpy(np.vstack(self.inputs).astype(np.float32))
        self.targets = torch.from_numpy(np.vstack(self.targets).astype(np.float32))
        self.rollout_ts = np.vstack(self.rollout_ts).astype(np.float32)
        self.prop_latent = np.vstack(self.prop_latent).astype(np.float32)

        # this computes the normalization
        if self.mode == 'train':
            self._preprocess_dataset()

        print('Found {} samples belonging to {} experiments:'.format(
            self.num_samples, self.num_experiments))

    def __len__(self):
        return self.num_samples

    def _add_features(self):
        self.input_prop_fts += ["rpy_0",
                                "rpy_1"]

        joint_dim = 12
        action_dim = 12
        latent_dim = self.config.latent_dim
        self.latent_dim = latent_dim

        for i in range(joint_dim):
            self.input_prop_fts.append("joint_angles_{}".format(i))
        for i in range(joint_dim):
            self.input_prop_fts.append("joint_vel_{}".format(i))
        for i in range(action_dim):
            self.input_prop_fts.append("last_action_{}".format(i))
        self.input_prop_fts.extend(["command_0", "command_1"])

        if self.config.input_use_depth:
            self.input_frame_fts = ["depth_frame_counter"]
        else:
            self.input_frame_fts = ["frame_counter"]
        self.input_fts = self.input_prop_fts + self.input_frame_fts
        self.target_fts = []
        for i in range(self.config.latent_dim):
            self.target_fts.append(f"prop_latent_{i}")

    def process_raw_latent(self, data, ts):

        data_freq = 90
        lookheads = self.config.lookhead
        self.num_lookheads = len(lookheads)

        predictive_latent = []
        smoothed_target = data
        for i in range(data.shape[1]):
            if (i != 8): # does not contain future geometry
                predictive_latent.append(np.expand_dims(savgol_filter(data[:, i], 41, 3),1))


        gamma1_with_lh = np.zeros_like(data[:,8])
        for k in range(len(lookheads)):
            # Advance only for future geometry, without any smoothing
            current_lookhead = int(lookheads[k]*data_freq)
            gamma1_with_lh = np.roll(smoothed_target[:,8],
                                  int(-current_lookhead))
            if current_lookhead >= 0:
                gamma1_with_lh[-current_lookhead:] = data[-current_lookhead:,8]
            else:
                gamma1_with_lh[:-current_lookhead] = data[:-current_lookhead, 8]
            predictive_latent.append(np.expand_dims(gamma1_with_lh,1))

        predictive_latent = np.hstack(predictive_latent)

        return predictive_latent

    def _decode_experiment(self, dir_subpath):
        propr_file = os.path.join(dir_subpath, "proprioception.csv")
        assert os.path.isfile(propr_file), "Not Found proprioception file"
        df_prop = pd.read_csv(propr_file, delimiter=',')

        current_img_paths = []
        if self.config.input_use_imgs:
            if self.config.input_use_depth:
                r_ext = '.tiff'
            else:
                r_ext = '.jpg'
            img_dir = os.path.join(dir_subpath, "img")
            for f in os.listdir(img_dir):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in [r_ext]:
                    continue
                current_img_paths.append(os.path.join(img_dir,f))
            if len(current_img_paths) == 0:
                raise IOError("Not found images")
        self.images_path.extend(sorted(current_img_paths))

        if self.debug:
            print("Average sampling frequency of proprioception is %.6f" % (
                        1.0 / np.mean(np.diff(np.unique(df_prop["time_from_start"].values)))))
        inputs, targets = df_prop[self.input_fts].values, df_prop[self.target_fts].values

        inputs[:,-1] += self.last_frame_idx


        ts = df_prop["time_from_start"]
        if self.config.input_use_depth:
            fc = df_prop["depth_frame_counter"]
        else:
            fc = df_prop["frame_counter"]

        prop_l = targets
        targets = self.process_raw_latent(prop_l, ts)

        self.inputs.append(inputs)
        self.targets.append(targets)

        final_idx = len(self.input_prop_fts)
        input_prop_features_v = inputs[:, :final_idx]

        if self.raw_prop_inputs is None:
            self.raw_prop_inputs = input_prop_features_v
            self.raw_latent_target = targets
        else:
            self.raw_prop_inputs = np.concatenate([self.raw_prop_inputs, input_prop_features_v], axis=0)
            self.raw_latent_target = np.concatenate([self.raw_latent_target, targets], axis=0)
        self.last_frame_idx += len(current_img_paths)
        self.num_samples += inputs.shape[0]
        #self.num_samples += len(idxs)
        self.rollout_boundaries.append(self.num_samples)
        self.rollout_ts.append(np.expand_dims(fc[:inputs.shape[0]], axis=1))
        self.prop_latent.append(prop_l)

    def _preprocess_dataset(self):
        if self.normalization_coeffs is None:
            self.input_prop_mean = np.mean(self.raw_prop_inputs, axis=0).astype(np.float32)
            self.input_prop_std = np.std(self.raw_prop_inputs, axis=0).astype(np.float32)
            self.target_latent_mean = np.mean(self.raw_latent_target, axis=0).astype(np.float32)
            self.target_latent_std = np.std(self.raw_latent_target, axis=0).astype(np.float32)
        else:
            self.input_prop_mean = self.normalization_coeffs[0]
            self.input_prop_std = self.normalization_coeffs[1]
            self.target_latent_mean = self.normalization_coeffs[2]
            self.target_latent_std = self.normalization_coeffs[3]

    def get_normalization(self):
        return self.input_prop_mean, self.input_prop_std, self.target_latent_mean, self.target_latent_std

    def setup_data_pipeline(self):
        if self.config.input_use_depth:
            self.preprocess_pipeline = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.CenterCrop((240,320)),
                                       ])
        else:
            self.preprocess_pipeline = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.CenterCrop((240,320)),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       ])

    def image_processing(self, fname):
        if self.config.input_use_depth:
            input_image = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
            input_array = np.asarray(input_image, dtype=np.float32)
            input_array = np.minimum(input_array, 4000)
            input_array = (input_array - 2000) / 2000
        else:
            input_image = cv2.imread(fname)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            input_array = np.asarray(input_image, dtype=np.float32)
        return input_array

    def preprocess_data(self, prop_numpy, img_numpy):
        prop = torch.from_numpy(prop_numpy.astype(np.float32))
        try:
            img_numpy = np.stack(img_numpy, axis=-1).astype(np.float32)
        except:
            print(img_numpy[0].shape)
            print(img_numpy[1].shape)
            print(img_numpy[2].shape)
        if self.config.input_use_depth:
            img_numpy = np.minimum(img_numpy, 4000)
            img_numpy = (img_numpy - 2000) / 2000
        img = self.preprocess_pipeline(img_numpy)
        return prop, img

    def __getitem__(self, idx):
        prop_data = np.zeros((self.config.history_len,len(self.input_prop_fts)), dtype=np.float32)
        start_idx = np.maximum(0, idx - self.config.history_len)
        actual_history_length = idx - start_idx
        if actual_history_length > 0:
            prop_data[-actual_history_length:] = self.inputs[start_idx:idx, :len(self.input_prop_fts)]
        else:
            prop_data[-1] = self.inputs[idx, :len(self.input_prop_fts)]

        target = self.targets[idx]
        frame_skip = self.config.frame_skip # take one every n
        if self.config.input_use_imgs:
            frame_idx_start = int(self.inputs[idx][-1].numpy())
            imgs = []
            for i in reversed(range(3)):
                frame_idx = np.maximum(0, frame_idx_start - i*frame_skip)
                frame_path = self.images_path[frame_idx]
                img = self.image_processing(frame_path)
                imgs.append(img)
            imgs = np.stack(imgs, axis=-1)
            imgs = self.preprocess_pipeline(imgs)

            return prop_data, imgs, target
        else:
            return prop_data, 0.0, target

if __name__ == "__main__":
    class DebugSettings:
        def __init__(self):
            self.input_use_imgs = True
            self.history_len = 20
            self.input_use_depth = False
            self.latent_size = 8
            self.lookhead = 0.5


    settings = DebugSettings()
    data = VisionPropDataset("/path/to/dataset/test", config=settings, mode='train', debug=True)
    sample = data.__getitem__(20)
