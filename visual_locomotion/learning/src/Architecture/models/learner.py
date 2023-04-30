from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import os
import random
import yaml
from tqdm import tqdm
import time
import json
try:
    import wandb
except:
    wandb = None


from Architecture.models.dataset import VisionPropDataset
from Architecture.models.arch_squeezenet import build_model
from Architecture.models.normalization import Normalization


class Learner:
    def __init__(self, config, mode='train') -> None:
        self.config = config
        self.mode = mode
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(True)

        seed = 10000  # np.random.randint(10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        ## parameters
        self.img_shape = (3, 240, 320)
        self.vector_dim = 40
        self.output_dim = self.config.latent_dim + len(self.config.lookhead) - 1
        cmd_dim = 2
        propr_dim = self.vector_dim - cmd_dim
        self.log_interval = 50
        self.log_loss = 0
        self.val_acc = 0
        self.best_val_loss = np.inf
        self.val_loss = np.inf
        self.val_nmse = np.inf
        self.epoch = 0

        self.network = build_model(prop_size=propr_dim,
                                   cmd_size=cmd_dim,
                                   img_shape=self.img_shape,
                                   pred_latent_shape=self.output_dim,
                                   history_len=self.config.history_len)

        if torch.cuda.is_available():
            self.network = self.network.cuda()

        if self.config.load_ckpt:
            assert os.path.isfile(self.config.ckpt_file), "Not found restore file"
            checkpoint = torch.load(self.config.ckpt_file)
            self.network.load_state_dict(checkpoint['model'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_loss = checkpoint['val_loss']
            self.val_nmse = checkpoint['val_nmse']
            self.loaded_ckpt = checkpoint

            print("------------------------------------------")
            print("Restored from {}".format(self.config.ckpt_file))
            print("------------------------------------------")
        else:
            print("------------------------------------------")
            print("Initializing network from scratch.")
            print("------------------------------------------")

        if wandb and self.mode == 'train':
            wandb.init(project="visual_locomotion", tags=config.tags,
                       settings=wandb.Settings(start_method="fork"),
                       name=self.config.name)
            wandb.config.update(config)
            if not os.path.isdir("save"):
                os.makedirs("save")

            config.wandb_name = wandb.run.name
            config.storage_folder = wandb.run.name
            # net file
            wandb.save("./models/arch_squeezenet.py")

        if self.mode != "test":
            self.prep_dataset = VisionPropDataset("", config, mode='deploy')
            self.network.eval()

    def write_summary_to_disk(self, norm_coeffs):
        yaml_dict = {}
        (means_in, stds_in, means_out, stds_out) = norm_coeffs
        yaml_dict["means_in"] = means_in.tolist()
        yaml_dict["stds_in"] = stds_in.tolist()
        yaml_dict["means_out"] = means_out.tolist()
        yaml_dict["stds_out"] = stds_out.tolist()

        fname_yaml = os.path.join(self.config.save_folder, "all.yaml")
        with open(fname_yaml, 'w') as f:
            s = yaml.dump(yaml_dict, f, default_flow_style=False)
        # add it to logging
        if wandb:
            wandb.save(fname_yaml)

    def train(self) -> None:

        train_dataset = VisionPropDataset(self.config.train_dir, self.config, mode='train')
        val_dataset = VisionPropDataset(self.config.val_dir, self.config, mode='test')
        train_data_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                       shuffle=True, num_workers=self.config.num_workers, pin_memory=True)
        val_data_loader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                                     shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=[int(0.3*len(train_data_loader)*self.config.epochs),
                                        int(0.7*len(train_data_loader)*self.config.epochs)],
            gamma=self.config.lr_decay_rate)

        if self.config.load_ckpt:
            self.optimizer.load_state_dict(self.loaded_ckpt['optimizer'])
            self.scheduler.load_state_dict(self.loaded_ckpt['scheduler'])

        #fall_criterion = nn.CrossEntropyLoss().cuda()
        latent_criterion = nn.MSELoss(reduction='none').cuda()
        self.normalization = Normalization(
                             train_data_loader.dataset.get_normalization())
        self.write_summary_to_disk(self.normalization.get_coeffs())
        self.network.train()

        log_loss = {'fall': 0, 'latent': 0}
        self.eval_interval = len(train_data_loader)

        for self.epoch in range(self.epoch, self.config.epochs):

            with tqdm(train_data_loader, total=len(train_data_loader)) as pbar:
                for batch_idx, data in enumerate(pbar):

                    step_ = self.epoch*len(train_data_loader) + batch_idx + 1
                    self.scheduler.step()

                    prop, img_emb, target = data
                    # norm prop
                    prop = self.normalization.normalize_inputs(prop)
                    norm_target_latent = self.normalization.normalize_labels(target)
                    if self.config.input_use_imgs:
                        output = self.network.forward(img_emb.cuda(), prop.cuda())
                    else:
                        output = self.network.forward(None, prop.cuda())
                    loss_latent = latent_criterion(output,
                                                   norm_target_latent.cuda())

                    loss = torch.mean(loss_latent)

                    try:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    except:
                        print("no gradients!")

                    log_loss['latent'] += loss.item()


                    if step_ % self.log_interval == 0:
                        self.curr_loss = (log_loss['latent']) / self.log_interval
                        postfix = {"epoch": self.epoch,
                                   "LR": self.scheduler.get_lr()[0],
                                   "Train MSE": log_loss['latent'] / self.log_interval,
                                   "Val Loss": self.val_loss,
                                   "Val (N)MSE": self.val_nmse}
                        pbar.set_postfix(postfix)
                        log_loss['latent'] = 0

                    if (step_ % self.eval_interval == 0):
                        self.network.eval()
                        print("Doing Evaluation")
                        val_loss = 0
                        nmse_loss = 0
                        eva_score = 0
                        mse_loss = 0
                        target_ls = []
                        pred_ls = []
                        for val_data in val_data_loader:
                            prop, img_emb, target = val_data
                            prop = self.normalization.normalize_inputs(prop)
                            norm_target_latent = self.normalization.normalize_labels(target)

                            with torch.no_grad():
                                if self.config.input_use_imgs:
                                    output = self.network.forward(img_emb.cuda(), prop.cuda())
                                else:
                                    output = self.network.forward(None, prop.cuda())
                                curr_nmse_loss = latent_criterion(output,
                                                                  norm_target_latent.cuda())
                                # normalized score
                                nmse_loss += torch.mean(curr_nmse_loss).item()

                                val_loss += torch.mean(curr_nmse_loss)

                                # compute other metrics
                                pred_latent = output.cpu().numpy()
                                pred_latent = self.normalization.unnormalize_labels(pred_latent)
                                # unnormalized score
                                latent_loss = latent_criterion(torch.tensor(pred_latent), target)
                                mse_loss += torch.mean(latent_loss).item()
                                eva_score += explained_variance_score(target,
                                                                      pred_latent,
                                                                      multioutput ='uniform_average')
                            # Append results
                            pred_ls.append(pred_latent)
                            target_ls.append(target.numpy())

                        # Log results at times
                        if (step_ % (2*self.eval_interval) == 0) and wandb:
                            target_ls = np.vstack(target_ls)
                            pred_ls = np.vstack(pred_ls)
                            ts = val_dataset.rollout_ts
                            stacked_data = np.hstack((ts,target_ls, pred_ls))

                            # Create plots
                            num_exps = val_dataset.num_experiments
                            columns = ['ts'] 
                            for i in range(self.output_dim):
                                columns.append(f'target_latent_{i}')
                            for i in range(self.output_dim):
                                columns.append(f'pred_latent_{i}')

                            for k in range(num_exps):
                                interval = slice(val_dataset.rollout_boundaries[k],
                                                 val_dataset.rollout_boundaries[k+1])
                                pd_frame = pd.DataFrame(data=stacked_data[interval],
                                                        columns=columns)
                                try:
                                    fig = plt.figure()
                                    for i in range(self.output_dim):
                                        fig = plt.figure()
                                        plt.plot(pd_frame['ts'], pd_frame[f'target_latent_{i}'], label = "Target")
                                        plt.plot(pd_frame['ts'], pd_frame[f'pred_latent_{i}'], label = "Prediction")
                                        plt.xlabel('Time')
                                        plt.legend()
                                        wandb.log({f"Exp {k}/Latent {i}": fig}, commit=False)
                                except:
                                    print("Could not draw figure")

                        self.val_loss = val_loss.cpu().numpy()/len(val_data_loader)
                        self.val_nmse = nmse_loss / len(val_data_loader)
                        self.val_mse = mse_loss / len(val_data_loader)
                        self.val_eva_score = eva_score / len(val_data_loader)

                        if wandb:

                            wandb.log({
                                'step': step_,
                                'epoch': self.epoch,
                                'Train loss': self.curr_loss,
                                'Val loss': self.val_loss,
                                "Learning Rate": self.scheduler.get_lr()[0],
                                'Val Normalized MSE': self.val_nmse,
                                'Val Unnormalized MSE': self.val_mse,
                                'Val Eva Score': self.val_eva_score
                            })

                        if(self.val_loss < self.best_val_loss):
                            self.best_val_loss = self.val_loss
                            save_file = os.path.join(
                                self.config.save_folder, 'best_model_'+str(wandb.run.name)+'.pth')
                            self.save_network(save_file)

                        save_file = os.path.join(self.config.save_folder,
                                                 'model_'+str(wandb.run.name) + "epoch_{}".format(self.epoch) + '.pth')
                        self.save_network(save_file)
                        # done evaluating
                        self.network.train()

        print("------------------------------")
        print("Training finished successfully")
        print("------------------------------")

    def save_network(self, save_file):
        state = {
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model': self.network.state_dict() if self.config.n_gpu <= 1 else self.network.module.state_dict(),
            'curr_loss': self.curr_loss,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'best_val_loss': self.best_val_loss,
            'val_nmse': self.val_nmse
        }
        torch.save(state, save_file)

    def initialize_network(self):
        '''
        Functions to init the graph at eval time
        '''
        # get normalization coefficients
        fname_yaml = os.path.join(Path(self.config.ckpt_file).parent, "all.yaml")
        if os.path.isfile(fname_yaml) and self.config.load_ckpt:
            print("Loading Normalization Coefficients")
            with open(fname_yaml, 'r') as f:
                yaml_list = yaml.load(f, Loader=yaml.SafeLoader)
                self.normalization = Normalization([np.array(yaml_list['means_in']),
                                                    np.array(yaml_list['stds_in']),
                                                    np.array(yaml_list['means_out']),
                                                    np.array(yaml_list['stds_out'])])

        else:
            print("Coefficients are not existing, cannot load")
            means_in = np.ones((1, self.vector_dim))
            stds_in = np.ones((1, self.vector_dim))
            means_out = np.ones((1, self.output_dim))
            stds_out = np.ones((1, self.output_dim))
            self.normalization = Normalization([means_in, stds_in,
                                                means_out, stds_out])

        print("------------------------------")
        print("Network Initialized           ")
        print("------------------------------")


    def getImageFts(self, img):
        '''
        Inference for real-time prediction
        '''
        prop = np.zeros((1, self.config.history_len, self.vector_dim))
        selected_frames = []
        for i in reversed(range(3)):
            selected_frames.append(img[-i*self.config.frame_skip-1])
        prop, img = self.prep_dataset.preprocess_data(prop, selected_frames)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            img_fts = self.network.forward_img(img.cuda())
        return img_fts

    def inference(self, input_dict):
        '''
        Inference for real-time prediction
        '''
        prop = input_dict['prop']
        prop = torch.from_numpy(prop.astype(np.float32))
        img_fts = input_dict['img_fts']
        assert prop.shape == (1, self.config.history_len, self.vector_dim)
        prop = self.normalization.normalize_inputs(prop)
        with torch.no_grad():
            if self.config.input_use_imgs or self.config.input_use_depth:
                predictions = self.network.forward(img_fts.cuda(), prop.cuda(), img_processed=True)
            else:
                predictions = self.network.forward(None, prop.cuda(), img_processed=True)
            # Return only the command
            pred_latent = np.squeeze(predictions.cpu().numpy())
            pred_latent = self.normalization.unnormalize_labels(pred_latent)
        return pred_latent
