import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualLocoNet(nn.Module):
    def __init__(self, prop_size=34, cmd_size=2, pred_latent_shape=3, history_len=20, use_prop=True):
        super(VisualLocoNet, self).__init__()
        model_ft = torch.hub.load('pytorch/vision:v0.10.0',
                                  'shufflenet_v2_x1_0', pretrained=True)
        modules=list(model_ft.children())[:-1]
        self.use_prop = use_prop
        self.prop_size = prop_size
        self.cmd_size = cmd_size
        self.history_len = history_len
        self.model_ft = nn.Sequential(*modules)
        self.fc1 = nn.Linear(288, 128)
        self.layer_reduce = nn.Conv2d(1024, 16, 1)
        self.layer_reduce_2 = nn.Conv2d(16, 2, 1)
        self.flatten =  nn.Flatten(start_dim = 1)
        self.fc2 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, pred_latent_shape)
        prop_input_size = prop_size + cmd_size
        self.output_prop_size = 128
        self.encoder = nn.Sequential(
            nn.Linear(prop_input_size, 32), nn.GELU(),
            )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.GELU(),
            nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.GELU(),
            nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.GELU(),
            nn.Flatten())
        self.linear_output = nn.Sequential(
            nn.Linear(128, self.output_prop_size), nn.GELU()
        )


    def forward_prop(self, prop):
        T = self.history_len
        bs = prop.shape[0]
        projection = self.encoder(prop.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output

    def forward_img(self, img):
        x = self.model_ft(img)
        x = F.gelu(x)
        x = self.layer_reduce(x)
        x = F.gelu(x)
        x = self.layer_reduce_2(x)
        x = F.gelu(x)
        x = self.flatten(x)
        return x

    def forward_fts(self, img_fts, prop_emb):
        if img_fts is None:
            img_fts = torch.zeros((prop_emb.shape[0], 288-prop_emb.shape[1])).cuda()
        x = torch.hstack((img_fts, prop_emb))
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc4(x)
        return x

    def forward(self, img, prop, img_processed=False):
        # This will clear out everything except IMU and the command
        prop[:, :, 2:-2] = 0.0
        prop_emb = self.forward_prop(prop)
        prop_emb = self.forward_prop(prop)
        if img is None:
            predictions = self.forward_fts(img_fts=None, prop_emb=prop_emb)
            return predictions
        if img_processed:
            img_fts = img
        else:
            img_fts = self.forward_img(img)
        predictions = self.forward_fts(img_fts=img_fts, prop_emb=prop_emb)
        return predictions
    

def build_model(prop_size=42, cmd_size=2, img_shape=(3,240,320), pred_latent_shape = 3, history_len=20, use_prop=True):
    model = VisualLocoNet(prop_size,cmd_size,pred_latent_shape, history_len=history_len, use_prop=use_prop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

if __name__ == '__main__':
    build_model()
