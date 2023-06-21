# 载入模块
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)

import pandas as pd
from tqdm import tqdm
import torchvision.utils as vutils
import torchvision.transforms as transforms
from datetime import datetime
import os
from moviepy.editor import VideoFileClip
import math
import json
from PIL import Image
import cv2
from cv2 import cv2


def get_data_info(file_path, csv_name):
    data_info_path = os.path.join(file_path, csv_name)
    data_info = pd.read_csv(data_info_path,
                            usecols=[
                                'Dialogue_ID', 'Utterance_ID', 'Speaker',
                                'Emotion', 'StartTime', 'EndTime'
                            ])
    data_info = data_info[data_info['Speaker'].isin(
        ['Joey', 'Ross', 'Rachel', 'Phoebe', 'Monica',
         'Chandler'])].reset_index()
    data_info['dur'] = None
    data_info['filename'] = ''
    for i in tqdm(range(len(data_info)),
                  desc='reading test data info:',
                  ncols=100):
        data_info['filename'][i] = 'dia' + str(
            data_info['Dialogue_ID'][i]) + '_utt' + str(
                data_info['Utterance_ID'][i]) + '.mp4'
        data_info['StartTime'][i] = data_info['StartTime'][i].split(',')[0]
        data_info['EndTime'][i] = data_info['EndTime'][i].split(',')[0]
        data_info['dur'][i] = (datetime.strptime(
            data_info['EndTime'][i], "%H:%M:%S") - datetime.strptime(
                data_info['StartTime'][i], "%H:%M:%S")).seconds + 1
    return data_info


def get_pic(file_path, input_file_name, video_folder_name, pic_folder_name,
            data_info, info_df_dict, num_frames):
    if num_frames is None:
        video = EncodedVideo.from_path(
            os.path.join(file_path, video_folder_name, input_file_name))
        clip = VideoFileClip(
            os.path.join(file_path, video_folder_name, input_file_name))
        clip_duration = math.ceil(clip.duration)
        if clip_duration < 1:
            pass
        for clip_start_sec in range(clip_duration - 1):
            cur_video_data = video.get_clip(
                start_sec=clip_start_sec,
                end_sec=clip_start_sec + 3)['video'].transpose(0,
                                                               1).unique(dim=0)
            cur_pic_data = cur_video_data[-1]
            # 复制
            input_tensor = cur_pic_data.clone().detach()
            # 到cpu
            input_tensor = input_tensor.to(torch.device('cpu'))

            save_filename = '%s-%d.jpg' % (input_file_name.split('.')[0],
                                           clip_start_sec)
            vutils.save_image(
                input_tensor,
                '%s/%s' %
                (os.path.join(file_path, pic_folder_name), save_filename),
                normalize=True)
            # tensor保存进csv
            cur_filename = save_filename.split('-')[0] + '.mp4'
            if cur_filename in data_info['filename'].values:
                info_df_dict['filename'].append(save_filename)
                info_df_dict['cur_filename'].append(cur_filename)
                info_df_dict['Speaker'].append(
                    data_info[data_info['filename'] ==
                              cur_filename]['Speaker'].values[0])
                info_df_dict['Emotion'].append(
                    data_info[data_info['filename'] ==
                              cur_filename]['Emotion'].values[0])

                totensor = transforms.ToTensor()
                img = totensor(input_tensor)
                # img = img.tolist()
                info_df_dict['img'].append(img)

    if num_frames is not None:
        current_file_list = os.listdir(os.path.join(file_path,
                                                    pic_folder_name))
        vc = cv2.VideoCapture(
            os.path.join(file_path, video_folder_name,
                         input_file_name))  #读入视频文件
        fps = int(vc.get(7))
        interval = fps // num_frames
        clip_num = []
        for i in range(num_frames):
            clip_num.append((i + 1) * interval)
        rval = vc.isOpened()  #判断视频是否打开 返回True或Flase
        c = 1
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()
            if rval:
                if c in clip_num:
                    save_filename = '%s-%d.jpg' % (
                        input_file_name.split('.')[0], clip_num.index(c))
                    if save_filename in current_file_list:
                        continue
                    # print(save_filename)
                    cv2.imwrite('%s/%s' % (os.path.join(
                        file_path, pic_folder_name), save_filename),
                                frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                    # tensor保存进csv
                    cur_filename = save_filename.split('-')[0] + '.mp4'
                    if cur_filename in data_info['filename'].values:
                        info_df_dict['filename'].append(save_filename)
                        info_df_dict['cur_filename'].append(cur_filename)
                        info_df_dict['Speaker'].append(
                            data_info[data_info['filename'] ==
                                      cur_filename]['Speaker'].values[0])
                        info_df_dict['Emotion'].append(
                            data_info[data_info['filename'] ==
                                      cur_filename]['Emotion'].values[0])

                        input_tensor = cv2.imread('%s/%s' % (os.path.join(
                            file_path, pic_folder_name), save_filename))
                        totensor = transforms.ToTensor()
                        img = totensor(input_tensor)
                        # img = img.tolist()
                        info_df_dict['img'].append(img)
                c = c + 1
        vc.release()


def main():
    file_path = '/workspace/chi149/MELD/MELD.Raw/test'
    csv_name = 'test_sent_emo.csv'
    video_folder_name = 'output_repeated_splits_test'
    pic_folder_name = 'test_pic1'
    info_csv_pkl = 'test_tensor_2pic.pkl'

    has_pic = True
    num_frames = 2

    data_info = get_data_info(file_path, csv_name)
    # len(data_info)

    info_df_dict = {}
    info_df_dict['filename'] = []
    info_df_dict['cur_filename'] = []
    info_df_dict['Speaker'] = []
    info_df_dict['Emotion'] = []
    info_df_dict['img'] = []
    if has_pic == False:
        for input_file_name in tqdm(os.listdir(
                os.path.join(file_path, video_folder_name)),
                                    desc='getting pic from test video:',
                                    ncols=100):
            if input_file_name in data_info['filename'].values:
                get_pic(file_path, input_file_name, video_folder_name,
                        pic_folder_name, data_info, info_df_dict, num_frames)
            # else:
            #     print(input_file_name)
            #     print('not in')
    elif has_pic == True:
        for save_filename in tqdm(
                os.listdir(os.path.join(file_path, pic_folder_name)),
                desc='getting pic tensor from test pictures:',
                ncols=100):
            # print(save_filename)
            # tensor保存进csv
            cur_filename = save_filename.split('-')[0] + '.mp4'
            # print(cur_filename)
            if cur_filename in data_info['filename'].values:
                info_df_dict['filename'].append(save_filename)
                info_df_dict['cur_filename'].append(cur_filename)
                info_df_dict['Speaker'].append(
                    data_info[data_info['filename'] ==
                              cur_filename]['Speaker'].values[0])
                info_df_dict['Emotion'].append(
                    data_info[data_info['filename'] ==
                              cur_filename]['Emotion'].values[0])

                img_path = os.path.join(file_path, pic_folder_name,
                                        save_filename)
                img = Image.open(img_path)
                img = transforms.ToTensor()(img)
                resize = transforms.Resize([224, 224])
                img = resize(img)
                info_df_dict['img'].append(img)
    print('getting pic tensor from test pictures end!')

    info_df = pd.DataFrame(info_df_dict)
    info_df.to_pickle(os.path.join(file_path, info_csv_pkl))
    # test_data_df1 = info_df[['cur_filename', 'Speaker', 'Emotion', 'img']]
    # test_data_df1 = info_df.sort_values(['filename'], ascending=True)[[
    #     'cur_filename', 'Speaker', 'Emotion', 'img'
    # ]].sort_values(['cur_filename'], ascending=True)
    # test_data_df1 = test_data_df1.reset_index(drop=True)
    # new_test_data1 = test_data_df1[['cur_filename', 'Speaker', 'Emotion']]
    # new_test_data1.drop_duplicates(inplace=True)
    # new_test_data1 = new_test_data1.reset_index(drop=True)
    # new_test_data1['pic_cnt'] = None
    # new_test_data1['img'] = None
    # test_videos = new_test_data1['cur_filename'].values
    # for video in tqdm(test_videos):
    #     imgs = test_data_df1[test_data_df1['cur_filename'] ==
    #                          video]['img'].values.tolist()

    #     index = new_test_data1[new_test_data1['cur_filename'] ==
    #                            video].index.tolist()[0]
    #     new_test_data1['pic_cnt'][index] = len(imgs)
    #     new_test_data1['img'][index] = torch.cat(imgs, dim=0)

    # new_test_data1.to_pickle(os.path.join(file_path, info_csv_pkl))


if __name__ == '__main__':
    main()