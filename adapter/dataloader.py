from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch


class getData(Dataset):
    def __init__(self, data, transform, type):
        # Transforms
        self.transform = transform
        self.type = type
        # read csv
        if self.type == 'Speaker':
            self.data_info = data[['img', 'Speaker']]
            # self.data_info = data[['new_img', 'Speaker']]
        elif self.type == 'Emotion':
            self.data_info = data[['img', 'Emotion']]
            # self.data_info = data[['new_img', 'Emotion']]
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])

    def __len__(self):
        return len(self.data_info)

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

        # get image
        single_img_tensor = self.image_arr[index]

        # get label
        single_image_label = label_dic[self.label_arr[index]]

        return (single_img_tensor, single_image_label)


# def compute_class_weights(data_df, target_column):
#     class_counts = data_df[target_column].value_counts()
#     print(class_counts)
#     total_samples = class_counts.sum()
#     print(total_samples)
#     print(len(class_counts))
#     class_weights = total_samples / (class_counts * len(class_counts))
#     print(class_weights.to_dict())
#     return class_weights.to_dict()


def getDataLoader(data_df, type, transforms, phase, batch_size=100):
    data = getData(
        data=data_df,
        transform=transforms,
        type=type,
    )

    dataloader = None

    if phase == 'train':
        class_counts = data_df[type].value_counts()
        weight_list = 1 / torch.Tensor(class_counts)
        if type == 'Emotion':
            print(class_counts)
            print(weight_list)
            sampler = WeightedRandomSampler(weight_list,
                                            len(data_df),
                                            replacement=True)
            dataloader = DataLoader(data,
                                    batch_size=batch_size,
                                    sampler=sampler,
                                    drop_last=False)
        elif type == 'Speaker':
            dataloader = DataLoader(data,
                                    batch_size=batch_size,
                                    drop_last=False,
                                    shuffle=True)

    elif phase == 'test':
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            drop_last=False,
        )

    return dataloader
