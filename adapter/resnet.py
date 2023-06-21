import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import copy
import argparse

from dataloader import getDataLoader


class Resnet():
    def __init__(self, model, dataloaders, device, optimizer, criterion):
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = 0

    def get_epoch(self, epoch):
        self.epoch = epoch

    # def train_test(self.model, self.epoch, self.dataloaders, self.device, self.optimizer, self.criterion):
    def train_test(self):
        model = self.model.to(self.device)
        train_epoch_loss = 0
        train_epoch_acc = 0
        test_epoch_loss = 0
        test_epoch_acc = 0
        samples = 0
        loss_sum = 0
        correct_sum = 0

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for index, (data, labels) in enumerate(self.dataloaders[phase]):
                X = torch.as_tensor(data).to(self.device)
                labels = torch.as_tensor(labels).to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y = model(X)
                    # print(y)
                    loss = self.criterion(y, labels)

                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    loss_sum += loss.item() * X.shape[
                        0]  # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                    samples += X.shape[0]
                    _, predicted = torch.max(y.data, 1)

                    correct_sum += (predicted == labels).sum().item()

                    # Print batch statistics every 50 batches
                    if index % 50 == 49 and phase == "train":
                        print("{}:{} - loss: {}, acc: {}".format(
                            self.epoch + 1, index + 1,
                            float(loss_sum) / float(samples),
                            float(correct_sum) / float(samples)))

            # Print epoch statistics
            if phase == 'train':
                train_epoch_acc = float(correct_sum) / float(samples)
                train_epoch_loss = float(loss_sum) / float(samples)
                print("epoch: {} - {} loss: {}, {} acc: {}".format(
                    self.epoch + 1, phase, train_epoch_loss, phase,
                    train_epoch_acc))
            elif phase == 'test':
                test_epoch_acc = float(correct_sum) / float(samples)
                test_epoch_loss = float(loss_sum) / float(samples)
                print("epoch: {} - {} loss: {}, {} acc: {}".format(
                    self.epoch + 1, phase, test_epoch_loss, phase,
                    test_epoch_acc))
        return train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc

def get_plot(x, y, output_file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.savefig(output_file_name)
