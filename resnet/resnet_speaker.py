import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import copy
import argparse
import os

from dataloader import getDataLoader


def train_test(model,
               epoch,
               dataloaders,
               device,
               optimizer,
               criterion,
               scheduler=None):
    model = model.to(device)
    train_epoch_loss = 0
    train_epoch_acc = 0
    test_epoch_loss = 0
    test_epoch_acc = 0

    for phase in ["train", "test"]:
        samples = 0
        loss_sum = 0
        correct_sum = 0
        if phase == "train":
            model.train()
        else:
            model.eval()

        for index, (data, labels) in enumerate(dataloaders[phase]):
            X = torch.as_tensor(data).to(device)
            labels = torch.as_tensor(labels).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                y = model(X)
                # print(y)
                loss = criterion(y, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.item() * X.shape[
                    0]  # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                samples += X.shape[0]
                _, predicted = torch.max(y.data, 1)
                if phase == 'train':
                    print(predicted)
                correct_sum += (predicted == labels).sum().item()
                # Print batch statistics every 50 batches
                if index % 50 == 49 and phase == "train":
                    print("{}:{} - loss: {}, acc: {}".format(
                        epoch + 1, index + 1,
                        float(loss_sum) / float(samples),
                        float(correct_sum) / float(samples)))
        # Print epoch statistics
        if phase == 'train':
            train_epoch_acc = float(correct_sum) / float(samples)
            train_epoch_loss = float(loss_sum) / float(samples)
            print("epoch: {} - {} loss: {}, {} acc: {}".format(
                epoch + 1, phase, train_epoch_loss, phase, train_epoch_acc))
        elif phase == 'test':
            test_epoch_acc = float(correct_sum) / float(samples)
            test_epoch_loss = float(loss_sum) / float(samples)
            print("epoch: {} - {} loss: {}, {} acc: {}".format(
                epoch + 1, phase, test_epoch_loss, phase, test_epoch_acc))
    return train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc


def get_plot(x, y, output_file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.savefig(output_file_name)


def main():
    # get para from command
    parser = argparse.ArgumentParser(description="input parameter")
    parser.add_argument('-epoches', '--epoches', type=int, default=100)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=100)
    parser.add_argument('-train', '--train_path', type=str)
    parser.add_argument('-test', '--test_path', type=str)

    args = parser.parse_args()

    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转（-10到+10度之间）
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
        transforms.RandomResizedCrop(224),  # 随机裁剪为固定大小（例如224x224）
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # 标准化图像张量
    ])

    trans_test = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转（-10到+10度之间）
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
        transforms.RandomResizedCrop(224),  # 随机裁剪为固定大小（例如224x224）
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # 标准化图像张量
    ])

    # generate data
    train_data_df = pd.read_pickle(args.train_path)
    train_dataloader = getDataLoader(train_data_df, 'Speaker', trans_train,
                                     'train', args.batch_size)
    test_data_df = pd.read_pickle(args.test_path)
    test_dataloader = getDataLoader(test_data_df, 'Speaker', trans_test,
                                    'test', args.batch_size)
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = models.resnet101(pretrained=True)
    old_conv1 = net.conv1
    net.conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=old_conv1.bias,
    )

    in_channel = net.fc.in_features
    net.fc = nn.Sequential(nn.Linear(in_channel, 1024), nn.ReLU(inplace=True),
                           nn.Dropout(0.5), nn.Linear(1024, 512),
                           nn.ReLU(inplace=True), nn.Dropout(0.5),
                           nn.Linear(512, 256), nn.ReLU(inplace=True),
                           nn.Dropout(0.5), nn.Linear(256, 6))

    # net = models.vgg16(pretrained=True)
    # add_classifier = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
    #                                nn.Dropout(0.5), nn.Linear(512, 256),
    #                                nn.ReLU(), nn.Dropout(0.5),
    #                                nn.Linear(256, 6))
    # net.classifier.add_module("add_linear",
    #                           add_classifier)  #在vgg16的classifier下面增加线性层

    optimizer = optim.SGD(net.parameters(),
                          lr=0.0001,
                          momentum=0.9,
                          weight_decay=0.001)
    # optimizer = optim.Adam(net.parameters())
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    epoch_list = []

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    # no_improvement_count = 0  # test acc no increase time, stop training

    for epoch in range(args.epoches):
        train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_test(
            net, epoch, dataloaders, device, optimizer, criterion, None)
        # print(optimizer.param_groups[0]["lr"])

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        epoch_list.append(epoch)

        # Deep copy the model
        if test_epoch_acc > best_acc:
            # no_improvement_count = 0
            best_acc = test_epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
        # elif test_epoch_acc <= best_acc:
        #     no_improvement_count += 1
        # print(no_improvement_count)
        # if no_improvement_count >= 40:
        #     break

    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.mkdir("./speaker{}".format(cur_time))
    torch.save(best_model_wts, "./speaker{}/best.pth".format(cur_time))
    get_plot(epoch_list, train_loss,
             './speaker{}/train_loss.jpg'.format(cur_time))
    get_plot(epoch_list, train_acc,
             './speaker{}/train_acc.jpg'.format(cur_time))
    get_plot(epoch_list, test_loss,
             './speaker{}/test_loss.jpg'.format(cur_time))
    get_plot(epoch_list, test_acc, './speaker{}/test_acc.jpg'.format(cur_time))
    # print(train_acc)
    # print(train_loss)
    # print(train_acc)
    # print(train_loss)


if __name__ == '__main__':
    main()
