import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import cv2
import torch
import pickle
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler


def get_pic_tensor(data_path, pic_folder, filename_dic, video_label_dic,
                   trans):
    video_tensor_arr = []
    label_arr = []

    for video_name in video_label_dic.keys():
        label = video_label_dic[video_name]

        pic_no = 0
        frames = []

        for name in filename_dic[video_name]:
            if name[-3:] != 'jpg':
                continue
            cur_pic_path = os.path.join(data_path, pic_folder, name)
            input_tensor = cv2.imread(cur_pic_path)
            img = trans(input_tensor)
            frames.append(img.numpy())
            pic_no += 1

        if len(frames) != 8 or pic_no != 8:
            print(video_name + ":" + str(len(frames)) + ':' + str(pic_no))
        else:
            transformed_video_tensor = torch.Tensor(frames)

        video_tensor_arr.append(transformed_video_tensor)
        label_arr.append(label)
    return video_tensor_arr, label_arr


class VideoData(Dataset):
    def __init__(self, data_path, pic_folder, filename_dic_pkl_name,
                 label_dic_pickle_name, trans, type):
        self.data_path = data_path
        self.pic_folder = pic_folder
        self.filename_dic_pkl_name = filename_dic_pkl_name
        self.label_dic_pickle_name = label_dic_pickle_name
        self.trans = trans
        self.type = type

        with open(os.path.join(self.data_path, self.filename_dic_pkl_name),
                  'rb') as f1:
            self.filename_dic = pickle.load(f1)
        with open(os.path.join(self.data_path, self.label_dic_pickle_name),
                  'rb') as f2:
            self.video_label_dic = pickle.load(f2)
        self.video_tensor_arr, self.label_arr = get_pic_tensor(
            self.data_path, self.pic_folder, self.filename_dic,
            self.video_label_dic, self.trans)
        self.label_count = dict(Counter(self.label_arr))

        if self.type == 'Speaker':
            self.label_dic = {
                'Joey': 0,
                'Ross': 1,
                'Rachel': 2,
                'Phoebe': 3,
                'Monica': 4,
                'Chandler': 5,
            }
        elif self.type == 'Emotion':
            self.label_dic = {
                'anger': 0,
                'disgust': 1,
                'fear': 2,
                'joy': 3,
                'sadness': 4,
                'surprise': 5,
                'neutral': 6,
            }

    def __len__(self):
        return len(self.filename_dic.keys())

    def __getitem__(self, index):
        cur_video_tensor = self.video_tensor_arr[index]
        cur_label = self.label_dic[self.label_arr[index]]

        return (cur_video_tensor, cur_label)


def getVideoDataLoader(data_path, pic_folder, filename_dic_pkl_name,
                       label_dic_pickle_name, trans, type, phase, batch_size):
    data = VideoData(data_path=data_path,
                     pic_folder=pic_folder,
                     filename_dic_pkl_name=filename_dic_pkl_name,
                     label_dic_pickle_name=label_dic_pickle_name,
                     trans=trans,
                     type=type)

    if type == 'Emotion' and phase == 'train':
        print
        print(data.label_count)
        train_data_len = len(data.label_arr)
        print(train_data_len)
        print(data.label_arr)
        label_count_list = []
        for label in [
                'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise',
                'neutral'
        ]:
            if label in list(data.label_count.keys()):
                label_count_list.append(data.label_count[label])
            else:
                label_count_list.append(0)
        print(label_count_list)
        class_weights = train_data_len / torch.tensor(label_count_list,
                                                      dtype=torch.float)
        print(class_weights)
        print(data.label_dic)
        data_weights = [
            class_weights[data.label_dic[w]] for w in data.label_arr
        ]
        sampler = WeightedRandomSampler(data_weights,
                                        num_samples=train_data_len,
                                        replacement=True)
        dataloader = DataLoader(data,
                                batch_size=batch_size,
                                drop_last=False,
                                sampler=sampler,
                                shuffle=False,
                                num_workers=16)

    else:
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            drop_last=False,
            # shuffle=True,
            num_workers=16)

    return dataloader