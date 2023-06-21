import torch.nn as nn
import torchvision.models as models


class Conv_LSTM(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_layers, bidirectional,
                 num_classes):
        super(Conv_LSTM, self).__init__()
        self.conv_model = Conv(latent_dim)
        self.lstm = LSTM(latent_dim, hidden_size, num_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(
                2 * hidden_size if bidirectional == True else hidden_size,
                num_classes), nn.Softmax(dim=-1))

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.lstm(lstm_input)[:, -1, :]
        output = self.output_layer(lstm_output)
        return output


class Conv(nn.Module):
    def __init__(self, latent_dim):
        super(Conv, self).__init__()
        self.conv = models.resnet152(pretrained=True)
        # 固定卷积层只训练最后fc
        for param in self.conv.parameters():
            param.requires_grad = False
        # 只训练最后的fc
        self.conv.fc = nn.Sequential(nn.Linear(self.conv.fc.in_features, 1024),
                                     nn.ReLU(inplace=True), nn.Dropout(0.5),
                                     nn.Linear(1024,
                                               512), nn.ReLU(inplace=True),
                                     nn.Dropout(0.5), nn.Linear(512, 256),
                                     nn.ReLU(inplace=True), nn.Dropout(0.5),
                                     nn.Linear(256, latent_dim))

    def forward(self, x):
        return self.conv(x)


class LSTM(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_layers, bidirectional):
        super(LSTM, self).__init__()
        self.LSTM = nn.LSTM(latent_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_size(self):
        self.hidden_state = None

    def forward(self, x):
        output, self.hidden_state = self.LSTM(x, self.hidden_state)
        return output
