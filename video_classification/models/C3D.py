import torch
import torch.nn as nn


class cnn3d(nn.Module):
    def __init__(self, num_classes):
        super(cnn3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 256, 3, 2, 1)
        self.conv2 = nn.Conv3d(256, 128, 3, 2, 1)
        self.conv3 = nn.Conv3d(128, 64, 2, 2, 1)
        self.conv4 = nn.Conv3d(64, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2)
        # self.pool2 = nn.MaxPool3d(2)
        # self.linear1 = nn.Linear(128*8,128)
        self.batchnorm = nn.BatchNorm3d(32)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(32 * 15 * 15, num_classes)

    def forward(self, x):
        batch_size, depth, channels, h_x, w_x = x.shape
        x = x.view(batch_size, channels, depth, h_x, w_x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm(x)
        x = x.view(x.size()[0], -1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.softmax(x)
        x = self.linear2(x)
        return x