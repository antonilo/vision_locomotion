import os
import yaml
import datetime
import shutil

def read_config(config_yaml, mode='train'):
    if mode == 'train':
        return trainConfig(config_yaml)
    if mode == 'test':
        return baseConfig(config_yaml)


class baseConfig:
    def __init__(self, config_yaml) -> None:
        assert os.path.isfile(config_yaml), config_yaml
        with open(config_yaml, 'r') as stream:
            config = yaml.safe_load(stream)
            self.config = config



            self.name = config['name']
            general = config['general']
            self.load_ckpt = general['load_ckpt']
            self.ckpt_file = general['ckpt_file']
            self.tags = general['tags']
            self.input_use_imgs = general['use_imgs']
            self.input_use_depth = general['use_depth']
            self.frame_skip = general['frame_skip']
            self.lookhead = general['lookhead']

            self.latent_dim = 10

            # arch stuff
            encoder = config['encoder']
            self.history_len = encoder['history_len']

            predictor = config['predictor']
            self.pred_hid_sizes = predictor['pred_hid_sizes']


        tags = self.tags.split(',')
        self.tags = list([])
        for it in tags:
            self.tags.append(it)

        self.n_gpu = 1  # torch.cuda.device_count()
        


class trainConfig(baseConfig):
    def __init__(self, config_yaml):
        super(trainConfig, self).__init__(config_yaml)
        self.add_flags()

        # some extras
        self.model_name = '{}'.format(self.tags[-1])

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_folder = os.path.join(self.save_path, current_time)
        self.save_folder = os.path.join(base_folder, self.model_name)
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
            config_copy_filepath = os.path.join(self.save_folder, 'config.yaml')
            shutil.copyfile(config_yaml, config_copy_filepath)

    def add_flags(self):
        train = self.config['train']
        self.train_dir = train['train_dir']
        self.val_dir = train['val_dir']
        self.test_dir = train['test_dir']
        assert os.path.isdir(self.train_dir), self.train_dir
        assert os.path.isdir(self.val_dir), self.val_dir
        self.batch_size = train['batch_size']
        self.save_path = os.path.join(train['save_path'], self.name)
        self.num_workers = train['num_workers']
        self.epochs = train['epochs']

        optimization = self.config['optimization']
        self.lr = optimization['lr']
        self.lr_decay_epochs = optimization['lr_decay_epochs']
        self.lr_decay_rate = optimization['lr_decay_rate']
        self.weight_decay = float(optimization['weight_decay'])
        self.momentum = optimization['momentum']

        # extra stuff
        iterations = self.lr_decay_epochs.split(',')
        self.lr_decay_epochs = list([])
        for it in iterations:
            self.lr_decay_epochs.append(int(it))
