import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet50
import torchvision.transforms as transforms

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import copy
import argparse

from dataloader import get_train_test_dataloader
from resnet import Resnet, get_plot


def main():
    # get para from command
    parser = argparse.ArgumentParser(description="input parameter")
    parser.add_argument('-epoches', '--epoches', type=int, default=100)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=100)
    parser.add_argument('-train', '--train_path', type=str)
    parser.add_argument('-test', '--test_path', type=str)
    parser.add_argument('-type', '--type', type=str)

    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose(
        [transforms.CenterCrop(224),
         transforms.ToTensor(), normalize])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate dataloader
    speaker_dataloaders = get_train_test_dataloader(args.train_path,
                                                    args.test_path, 'Speaker',
                                                    trans, args.batch_size)
    emotion_dataloaders = get_train_test_dataloader(args.train_path,
                                                    args.test_path, 'Emotion',
                                                    trans, args.batch_size)

    # define speaker network
    speaker_net = models.resnet50()
    # speaker_net.fc = nn.Sequential(
    #     nn.Linear(speaker_net.fc.in_features, 256),
    #     nn.Sigmoid(),
    #     nn.Linear(256, 6),
    # )
    speaker_optimizer = optim.SGD(speaker_net.parameters(),
                                  lr=1e-3,
                                  momentum=0.9)
    speaker_criterion = nn.CrossEntropyLoss()
    speaker_resnet = Resnet(speaker_net, speaker_dataloaders, device,
                            speaker_optimizer, speaker_criterion)

    # speaker_best_model_wts = copy.deepcopy(speaker_net.state_dict())
    # speaker_best_acc = 0.0

    # define emotion network
    emotion_net = models.resnet50()
    # emotion_net.fc = nn.Sequential(
    #     nn.Linear(emotion_net.fc.in_features, 256),
    #     nn.Sigmoid(),
    #     nn.Linear(256, 7),
    # )
    emotion_optimizer = optim.SGD(emotion_net.parameters(),
                                  lr=1e-3,
                                  momentum=0.9)
    emotion_criterion = nn.CrossEntropyLoss()
    emotion_resnet = Resnet(emotion_net, emotion_dataloaders, device,
                            emotion_optimizer, emotion_criterion)

    # emotion_best_model_wts = copy.deepcopy(emotion_net.state_dict())
    # emotion_best_acc = 0.0

    # train the two network
    speaker_train_loss = []
    speaker_train_acc = []
    speaker_test_loss = []
    speaker_test_acc = []

    # emotion_train_loss = []
    # emotion_train_acc = []
    # emotion_test_loss = []
    # emotion_test_acc = []

    epoch_list = []

    # train the two model
    for epoch in range(args.epoches):
        speaker_resnet.epoch = epoch
        emotion_resnet.epoch = epoch

        speaker_train_epoch_loss, speaker_train_epoch_acc, speaker_test_epoch_loss, speaker_test_epoch_acc = speaker_resnet.train_test(
        )
        emotion_train_epoch_loss, emotion_train_epoch_acc, emotion_test_epoch_loss, emotion_test_epoch_acc = emotion_resnet.train_test(
        )

    # adapter
    input_adapter_size = speaker_resnet.model.fc.out_features + emotion_resnet.model.fc.out_features
    if args.type == 'speaker':
        output_adapter_size = emotion_net.fc.in_features
    elif args.type == 'emotion':
        output_adapter_size = emotion_net.fc.in_features
    adapter = nn.Linear(input_adapter_size, output_adapter_size)
    adapter = adapter.to(device)

    adapter_optimizer = optim.Adam(adapter.parameters(), lr=1e-3)
    adapter_criterion = nn.CrossEntropyLoss()

    # # add layers fc layer
    # speaker_resnet.model.fc = nn.Sequential(
    #     nn.Linear(speaker_resnet.model.fc.in_features, 256),
    #     nn.Sigmoid(),
    #     nn.Linear(256, 6),
    # )

    # emotion_resnet.model.fc = nn.Sequential(
    #     nn.Linear(emotion_resnet.model.fc.in_features, 256),
    #     nn.Sigmoid(),
    #     nn.Linear(256, 7),
    # )

    # adapter training
    for epoch in range(args.epoches):
        speaker_train_epoch_loss = 0
        speaker_train_epoch_acc = 0
        speaker_test_epoch_loss = 0
        speaker_test_epoch_acc = 0
        speaker_samples = 0
        speaker_loss_sum = 0
        speaker_correct_sum = 0

        # emotion_train_epoch_loss = 0
        # emotion_train_epoch_acc = 0
        # emotion_test_epoch_loss = 0
        # emotion_test_epoch_acc = 0
        # emotion_samples = 0
        # emotion_loss_sum = 0
        # emotion_correct_sum = 0

        for phase in ["train", "test"]:
            if phase == "train":
                adapter.train()
            else:
                adapter.eval()

            for index, data in enumerate(
                    zip(speaker_dataloaders[phase],
                        emotion_dataloaders[phase])):
                (speaker_X, speaker_labels), (emotion_X, emotion_labels) = data
                speaker_X = torch.as_tensor(speaker_X).to(device)
                speaker_labels = torch.as_tensor(speaker_labels).to(device)

                emotion_X = torch.as_tensor(emotion_X).to(device)
                emotion_labels = torch.as_tensor(emotion_labels).to(device)

                adapter_optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.no_grad():
                        out_speaker = speaker_resnet.model(speaker_X)
                    with torch.no_grad():
                        out_emotion = emotion_resnet.model(emotion_X)

                    out_combined = torch.cat((out_speaker, out_emotion), dim=1)
                    out_adapter = adapter(out_combined)
                    speaker_y = speaker_resnet.model.fc(out_adapter)
                    # emotion_y = emotion_resnet.model(emotion_y)

                    speaker_loss = adapter_criterion(speaker_y, speaker_labels)
                    # emotion_loss = criterion(emotion_y, emotion_labels)

                    if phase == "train":
                        speaker_loss.backward()
                        speaker_optimizer.step()

                        # emotion_loss.backward()
                        # emotion_optimizer.step()

                    speaker_loss_sum += speaker_loss.item() * speaker_X.shape[
                        0]  # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                    speaker_samples += speaker_X.shape[0]
                    _, speaker_predicted = torch.max(speaker_y.data, 1)
                    speaker_correct_sum += (
                        speaker_predicted == speaker_labels).sum().item()

                    # emotion_loss_sum += emotion_loss.item() * emotion_X.shape[0]  # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                    # emotion_samples += emotion_X.shape[0]
                    # _, emotion_predicted = torch.max(emotion_y.data, 1)
                    # emotion_correct_sum += (emotion_predicted == emotion_labels).sum().item()

                    # Print batch statistics every 50 batches
                    if index % 50 == 49 and phase == "train":
                        print("Speaker {}:{} - loss: {}, acc: {}".format(
                            epoch + 1, index + 1,
                            float(speaker_loss_sum) / float(speaker_samples),
                            float(speaker_correct_sum) /
                            float(speaker_samples)))

                    # if index % 50 == 49 and phase == "train":
                    #     print("Emotion {}:{} - loss: {}, acc: {}".format(
                    #         epoch + 1, index + 1,
                    #         float(emotion_loss_sum) / float(emotion_samples),
                    #         float(emotion_correct_sum) / float(emotion_samples)))
            # Print epoch statistics
            if phase == 'train':
                speaker_train_epoch_acc = float(speaker_correct_sum) / float(
                    speaker_samples)
                speaker_train_epoch_loss = float(speaker_loss_sum) / float(
                    speaker_samples)
                print("epoch: {} - {} loss: {}, {} acc: {}".format(
                    epoch + 1, phase, speaker_train_epoch_loss, phase,
                    speaker_train_epoch_acc))

                # emotion_train_epoch_acc = float(emotion_correct_sum) / float(emotion_samples)
                # emotion_train_epoch_loss = float(emotion_loss_sum) / float(emotion_samples)
                # print("epoch: {} - {} loss: {}, {} acc: {}".format(
                #     epoch + 1, phase, emotion_train_epoch_loss, phase, emotion_train_epoch_acc))
            elif phase == 'test':
                speaker_test_epoch_acc = float(speaker_correct_sum) / float(
                    speaker_samples)
                speaker_test_epoch_loss = float(speaker_loss_sum) / float(
                    speaker_samples)
                print("epoch: {} - {} loss: {}, {} acc: {}".format(
                    epoch + 1, phase, speaker_test_epoch_loss, phase,
                    speaker_test_epoch_acc))

                # emotion_test_epoch_acc = float(emotion_correct_sum) / float(emotion_samples)
                # emotion_test_epoch_loss = float(emotion_loss_sum) / float(emotion_samples)
                # print("epoch: {} - {} loss: {}, {} acc: {}".format(
                #     epoch + 1, phase, emotion_test_epoch_loss, phase, emotion_test_epoch_acc))

        # Deep copy the model
        # if speaker_test_epoch_acc > speaker_best_acc:
        #     speaker_best_acc = speaker_test_epoch_acc
        #     speaker_best_model_wts = copy.deepcopy(
        #         speaker_resnet.model.state_dict())

        # if emotion_test_epoch_acc > emotion_best_acc:
        #     emotion_best_acc = emotion_test_epoch_acc
        #     emotion_best_model_wts = copy.deepcopy(
        #         emotion_resnet.model.state_dict())

        speaker_train_loss.append(speaker_train_epoch_loss)
        speaker_train_acc.append(speaker_train_epoch_acc)
        speaker_test_loss.append(speaker_test_epoch_loss)
        speaker_test_acc.append(speaker_test_epoch_acc)

        # emotion_train_loss.append(emotion_train_epoch_loss)
        # emotion_train_acc.append(emotion_train_epoch_acc)
        # emotion_test_loss.append(emotion_test_epoch_loss)
        # emotion_test_acc.append(emotion_test_epoch_acc)
        epoch_list.append(epoch)

    # torch.save(
    #     speaker_best_model_wts, "./speaker_resnet50_{}.pth".format(
    #         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    get_plot(
        epoch_list, speaker_train_loss, './speaker_train_loss{}.jpg'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    get_plot(
        epoch_list, speaker_train_acc, './speaker_train_acc{}.jpg'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    get_plot(
        epoch_list, speaker_test_loss, './speaker_test_loss{}.jpg'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    get_plot(
        epoch_list, speaker_test_acc, './speaker_test_acc{}.jpg'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # torch.save(
    #     emotion_best_model_wts, "./emotion_resnet50_{}.pth".format(
    #         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # get_plot(
    #     epoch_list, emotion_train_loss, './emotion_train_loss{}.jpg'.format(
    #         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # get_plot(
    #     epoch_list, emotion_train_acc, './emotion_train_acc{}.jpg'.format(
    #         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # get_plot(
    #     epoch_list, emotion_test_loss, './emotion_test_loss{}.jpg'.format(
    #         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # get_plot(
    #     epoch_list, emotion_test_acc, './emotion_test_acc{}.jpg'.format(
    #         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == '__main__':
    main()
