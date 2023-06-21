from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import re
import os
import cv2
import torch
import pickle


def get_pic_tensor(data_path, pic_folder, filename_dic, video_label_dic,
                   trans):
    video_tensor_arr = []
    label_arr = []
    for video_name in video_label_dic.keys():
        label = video_label_dic[video_name]

        pic_no = 0
        frames = []

        for name in filename_dic[video_name]:
            cur_pic_path = os.path.join(data_path, pic_folder, name)
            input_tensor = cv2.imread(cur_pic_path)
            img = trans(input_tensor)
            frames.append(img.numpy())
            pic_no += 1

        if len(frames) != 15 or pic_no != 15:
            print(video_name + ":" + str(len(frames)) + ':' + str(pic_no))
            pass
        else:
            transformed_video_tensor = torch.Tensor(frames)

        video_tensor_arr.append(transformed_video_tensor)
        label_arr.append(label)
    print(len(video_tensor_arr))
    print(label_arr)
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
        print(len(self.filename_dic.keys()))
        with open(os.path.join(self.data_path, self.label_dic_pickle_name),
                  'rb') as f2:
            self.video_label_dic = pickle.load(f2)
        print(len(self.video_label_dic.keys()))

    def __len__(self):
        return len(self.filename_dic.keys())

    def __getitem__(self, index):
        if self.type == 'Speaker':
            label_dic = {
                'Joey': 0,
                'Ross': 1,
                'Rachel': 2,
                'Phoebe': 3,
                'Monica': 4,
                'Chandler': 5,
            }
        elif self.type == 'Emotion':
            label_dic = {
                'anger': 0,
                'disgust': 1,
                'fear': 2,
                'joy': 3,
                'sadness': 4,
                'surprise': 5,
                'neutral': 6,
            }
        self.video_tensor_arr, self.label_arr = get_pic_tensor(
            self.data_path, self.pic_folder, self.filename_dic,
            self.video_label_dic, self.trans)
        print(len(self.video_tensor_arr))
        print(len(self.label_arr))
        print(index)
        cur_video_tensor = self.video_tensor_arr[index]
        cur_label = label_dic[self.label_arr[index]]

        return (cur_video_tensor, cur_label)


def getVideoDataLoader(data_path, pic_folder, filename_dic_pkl_name,
                       label_dic_pickle_name, trans, type, batch_size):
    data = VideoData(data_path=data_path,
                     pic_folder=pic_folder,
                     filename_dic_pkl_name=filename_dic_pkl_name,
                     label_dic_pickle_name=label_dic_pickle_name,
                     trans=trans,
                     type=type)

    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            drop_last=False,
                            shuffle=True)

    return dataloader