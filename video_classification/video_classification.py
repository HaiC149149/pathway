import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import copy
import argparse
import os
from dataloader_video import getVideoDataLoader

from models.Resnet_3D import ResNet
from models.C3D import cnn3d

# model para
latent_dim = 512
hidden_size = 256
num_layers = 4
bidirectional = True
num_classes = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test(model,
               epoch,
               dataloaders,
               device,
               optimizer,
               criterion,
               type,
               scheduler=None):
    model = model.to(device)
    train_epoch_loss = 0
    train_epoch_acc = 0
    test_epoch_loss = 0
    test_epoch_acc = 0
    label_count = {}
    # 异常检测启动
    torch.autograd.set_detect_anomaly(True)

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

            if phase == 'train' and type == 'Emotion':
                unique, counts = np.unique(labels.cpu().numpy(),
                                           return_counts=True)
                # print(unique)
                # print(counts)
                for label in unique:
                    if label not in list(label_count.keys()):
                        label_count[label] = counts[unique.tolist().index(
                            label)]
                    else:
                        label_count[label] += counts[unique.tolist().index(
                            label)]

            with torch.set_grad_enabled(phase == 'train'):
                y = model(X)
                # print(y)
                # print(labels)
                # print(y.shape)
                # print(labels.shape)

                loss = criterion(y, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.item() * X.shape[
                    0]  # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                samples += X.shape[0]
                _, predicted = torch.max(y.data, 1)

                correct_sum += (predicted == labels).sum().item()
                # Print batch statistics every 50 batches
                if index % 50 == 49 and phase == "train":
                    print("{}:{} - loss: {}, acc: {}".format(
                        epoch + 1, index + 1,
                        float(loss_sum) / float(samples),
                        float(correct_sum) / float(samples)))

        # if scheduler is not None and phase == 'train':
        #     scheduler.step()
        # Print epoch statistics

        if phase == 'train':
            print(label_count)
            train_epoch_acc = float(correct_sum) / float(samples)
            train_epoch_loss = float(loss_sum) / float(samples)
            print("epoch: {} - {} loss: {}, {} acc: {}".format(
                epoch + 1, phase, train_epoch_loss, phase, train_epoch_acc))

        elif phase == 'test':
            test_epoch_acc = float(correct_sum) / float(samples)
            test_epoch_loss = float(loss_sum) / float(samples)
            print("epoch: {} - {} loss: {}, {} acc: {}".format(
                epoch + 1, phase, test_epoch_loss, phase, test_epoch_acc))

    # 异常检测关闭
    torch.autograd.set_detect_anomaly(False)

    return train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc


def get_plot(x, y, output_file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.savefig(output_file_name)


def main():
    # ------------------ get args ------------------
    parser = argparse.ArgumentParser(description="input parameter")
    parser.add_argument('-epoches', '--epoches', type=int)
    parser.add_argument('-batch_size', '--batch_size', type=int)
    parser.add_argument('-type', '--type', type=str)
    args = parser.parse_args()

    # ------------------ file path para ------------------
    train_data_path = '/workspace/chi149/MELD/MELD.Raw/train'
    train_pic_folder = 'train_video_pic8'
    train_filename_dic_pkl_name = 'train_video_filename_dic_8.pkl'

    test_data_path = '/workspace/chi149/MELD/MELD.Raw/test'
    test_pic_folder = 'test_video_pic8'
    test_filename_dic_pkl_name = 'test_video_filename_dic_8.pkl'

    if args.type == 'Speaker':
        train_label_dic_pickle_name = 'train_video_speaker_dic_8.pkl'
        test_label_dic_pickle_name = 'test_video_speaker_dic_8.pkl'
    elif args.type == 'Emotion':
        train_label_dic_pickle_name = 'train_video_emotion_dic_8.pkl'
        test_label_dic_pickle_name = 'test_video_emotion_dic_8.pkl'

    # train_data_path = '/workspace/chi149/MELD/pathway/video_classification/testdata'
    # train_pic_folder = 'train'
    # train_filename_dic_pkl_name = 'train_filename_dic.pkl'

    # test_data_path = '/workspace/chi149/MELD/pathway/video_classification/testdata'
    # test_pic_folder = 'test'
    # test_filename_dic_pkl_name = 'test_filename_dic.pkl'

    # train_label_dic_pickle_name = 'train_video_emotion.pkl'
    # test_label_dic_pickle_name = 'test_video_emotion.pkl'

    # ------------------ transforms ------------------
    trans_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ------------------ get dataloader ------------------
    print("------------------ get dataloader start ------------------")
    train_dataloader = getVideoDataLoader(train_data_path, train_pic_folder,
                                          train_filename_dic_pkl_name,
                                          train_label_dic_pickle_name,
                                          trans_train, args.type, 'train',
                                          args.batch_size)
    test_dataloader = getVideoDataLoader(test_data_path, test_pic_folder,
                                         test_filename_dic_pkl_name,
                                         test_label_dic_pickle_name,
                                         trans_test, args.type, 'test',
                                         args.batch_size)
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}

    print("train dataloader length: " + str(len(train_dataloader)))
    print("test dataloader length: " + str(len(test_dataloader)))
    print("------------------ get dataloader finish ------------------")

    # ------------------ model define ------------------
    # net = Conv_LSTM(latent_dim, hidden_size, num_layers, bidirectional,
    #                 num_classes)
    # net = Resnet.generate_model(101,num_classes)
    net = cnn3d(num_classes)
    net = net.to(device)

    optimizer = optim.SGD(net.parameters(),
                          lr=1e-3,
                          momentum=0.9,
                          weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    epoch_list = []

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for epoch in range(args.epoches):
        train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_test(
            net, epoch, dataloaders, device, optimizer, criterion, args.type,
            None)
        # scheduler.step(test_epoch_loss)
        print(optimizer.param_groups[0]["lr"])
        # Deep copy the model
        if test_epoch_acc > best_acc:
            best_acc = test_epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        epoch_list.append(epoch)

    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.mkdir("./result/emotion{}".format(cur_time))
    torch.save(best_model_wts, "./result/emotion{}/best.pth".format(cur_time))
    # torch.save(
    #     best_model_wts, "./emotion{}/vgg16.pth".format(cur_time))
    get_plot(epoch_list, train_loss,
             './result/emotion{}/train_loss.jpg'.format(cur_time))
    get_plot(epoch_list, train_acc,
             './result/emotion{}/train_acc.jpg'.format(cur_time))
    get_plot(epoch_list, test_loss,
             './result/emotion{}/test_loss.jpg'.format(cur_time))
    get_plot(epoch_list, test_acc,
             './result/emotion{}/test_acc.jpg'.format(cur_time))
    # print(train_acc)
    # print(train_loss)
    # print(train_acc)
    # print(train_loss)


if __name__ == '__main__':
    main()
