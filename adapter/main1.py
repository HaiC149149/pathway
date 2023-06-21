import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet152
import torchvision.transforms as transforms

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import copy
import argparse
import os

from dataloader import getDataLoader
from resnet import Resnet, get_plot
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


def main():
    # get para from command
    parser = argparse.ArgumentParser(description="input parameter")
    parser.add_argument('-epoches', '--epoches', type=int, default=100)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=100)
    parser.add_argument('-type', '--type', type=str, default='Speaker')

    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose(
        [transforms.CenterCrop(224),
         transforms.ToTensor(), normalize])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = '/workspace/chi149/MELD/MELD.Raw/train/train_tensor1.pkl'
    test_path = '/workspace/chi149/MELD/MELD.Raw/test/test_tensor_2pic.pkl'

    # generate dataloader
    train_data_df = pd.read_pickle(train_path)
    test_data_df = pd.read_pickle(test_path)
    train_speaker_dataloader = getDataLoader(train_data_df, 'Speaker', trans,
                                             'train', args.batch_size)
    test_speaker_dataloader = getDataLoader(test_data_df, 'Speaker', trans,
                                            'test', args.batch_size)

    train_emotion_dataloader = getDataLoader(train_data_df, 'Emotion', trans,
                                             'train', args.batch_size)
    test_emotion_dataloader = getDataLoader(test_data_df, 'Emotion', trans,
                                            'test', args.batch_size)

    speaker_dataloaders = {
        'train': train_speaker_dataloader,
        'test': test_speaker_dataloader
    }
    emotion_dataloaders = {
        'train': train_emotion_dataloader,
        'test': test_emotion_dataloader
    }

    # define speaker network
    speaker_net = models.resnet50(pretrained=True)
    speaker_model_path = '/workspace/chi149/MELD/pathway/resnet/speaker2023-05-23 17:05:44/best.pth'
    speaker_net.load_state_dict(torch.load(speaker_model_path), False)
    speaker_net.fc = nn.Sequential(
        nn.Linear(speaker_net.fc.in_features, 256),
        nn.Sigmoid(),
        nn.Linear(256, 6),
    )
    speaker_net = speaker_net.to(device)
    speaker_optimizer = optim.SGD(speaker_net.parameters(),
                                  lr=1e-3,
                                  momentum=0.9)
    speaker_criterion = nn.CrossEntropyLoss()
    speaker_resnet = Resnet(speaker_net, speaker_dataloaders, device,
                            speaker_optimizer, speaker_criterion)

    # speaker_best_model_wts = copy.deepcopy(speaker_net.state_dict())
    speaker_best_acc = 0.0

    # define emotion network
    emotion_net = models.resnet50(pretrained=True)
    emotion_model_path = '/workspace/chi149/MELD/pathway/resnet/emotion2023-05-24 11:36:37/best.pth'
    emotion_net.load_state_dict(torch.load(emotion_model_path), False)
    emotion_net.fc = nn.Sequential(
        nn.Linear(emotion_net.fc.in_features, 256),
        nn.Sigmoid(),
        nn.Linear(256, 7),
    )
    emotion_net = emotion_net.to(device)
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
    # for epoch in range(args.epoches):
    #     speaker_resnet.epoch = epoch
    #     emotion_resnet.epoch = epoch

    # speaker_train_epoch_loss, speaker_train_epoch_acc, speaker_test_epoch_loss, speaker_test_epoch_acc = speaker_resnet.train_test(
    # )
    # emotion_train_epoch_loss, emotion_train_epoch_acc, emotion_test_epoch_loss, emotion_test_epoch_acc = emotion_resnet.train_test(
    # )

    # adapter
    input_adapter_size = speaker_resnet.model.fc[
        0].in_features + emotion_resnet.model.fc[0].in_features
    # print(input_adapter_size)

    if args.type == 'speaker':
        output_adapter_size = speaker_net.fc[0].in_features
    elif args.type == 'emotion':
        output_adapter_size = emotion_net.fc[0].in_features

    # define adapter
    adapter = nn.Sequential(nn.Linear(input_adapter_size, 2048), nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(2048, output_adapter_size))
    adapter = adapter.to(device)

    adapter_optimizer = optim.Adam(adapter.parameters(), lr=1e-3)
    adapter_criterion = nn.CrossEntropyLoss()

    # adapter training
    for epoch in range(args.epoches):
        speaker_train_epoch_loss = 0
        speaker_train_epoch_acc = 0
        speaker_test_epoch_loss = 0
        speaker_test_epoch_acc = 0

        # emotion_train_epoch_loss = 0
        # emotion_train_epoch_acc = 0
        # emotion_test_epoch_loss = 0
        # emotion_test_epoch_acc = 0

        for phase in ["train", "test"]:
            speaker_samples = 0
            speaker_loss_sum = 0
            speaker_correct_sum = 0
            # emotion_samples = 0
            # emotion_loss_sum = 0
            # emotion_correct_sum = 0

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
                        return_layers = {'avgpool': 'avgpool'}
                        mid_getter_speaker = MidGetter(
                            speaker_resnet.model,
                            return_layers=return_layers,
                            keep_output=False)
                        out_speaker = mid_getter_speaker(
                            speaker_X)[0]['avgpool']
                    with torch.no_grad():
                        return_layers = {'avgpool': 'avgpool'}
                        mid_getter_emotion = MidGetter(
                            emotion_resnet.model,
                            return_layers=return_layers,
                            keep_output=False)
                        out_emotion = mid_getter_emotion(
                            emotion_X)[0]['avgpool']

                    out_combined = torch.cat((out_speaker, out_emotion), dim=1)
                    out_combined = out_combined.view(-1, input_adapter_size)
                    # print(out_combined.shape)

                    out_adapter = adapter(out_combined)

                    speaker_y = speaker_resnet.model.fc(out_adapter)
                    # emotion_y = emotion_resnet.model.fc(emotion_y)

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
                        print("{} {}:{} - loss: {}, acc: {}".format(
                            args.type, epoch + 1, index + 1,
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
        if speaker_test_epoch_acc > speaker_best_acc:
            speaker_best_acc = speaker_test_epoch_acc
            speaker_best_model_wts = copy.deepcopy(
                speaker_resnet.model.state_dict())

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

    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.mkdir("./result/{}_{}".format(args.type, cur_time))
    torch.save(speaker_best_model_wts,
               "./result/{}_{}/best.pth".format(args.type, cur_time))
    get_plot(
        epoch_list, speaker_train_loss,
        './result/{}_{}/adapter_train_loss.jpg'.format(args.type, cur_time))
    get_plot(
        epoch_list, speaker_train_acc,
        './result/{}_{}/adapter_train_acc.jpg'.format(args.type, cur_time))
    get_plot(
        epoch_list, speaker_test_loss,
        './result/{}_{}/adapter_test_loss.jpg'.format(args.type, cur_time))
    get_plot(epoch_list, speaker_test_acc,
             './result/{}_{}/adapter_test_acc.jpg'.format(args.type, cur_time))

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