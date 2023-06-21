import pandas as pd
import os
import re
from tqdm import tqdm
import cv2
import pickle


# get video tensor + save picture
def getVideoTensor(data_path, folder_name, video_name, save_folder_name,
                   video_info_df):
    if video_name[0] == '_' or video_name[
            0:5] == 'final' or video_name[-4:] != '.mp4' or video_name == '':
        return 'Other', 'Other', []
    dia_num = int(re.sub(r'\D', "", video_name[:-4].split('_')[0]))
    utt_num = int(re.sub(r'\D', "", video_name[:-4].split('_')[1]))
    cur_video_info = video_info_df[(video_info_df['Dialogue_ID'] == dia_num) &
                                   (video_info_df['Utterance_ID'] == utt_num)]
    if len(cur_video_info) == 0:
        return 'Other', 'Other', []
    else:
        cur_video_speaker = cur_video_info['Speaker'].values[0]
        cur_video_emotion = cur_video_info['Emotion'].values[0]
    if cur_video_speaker not in ('Joey','Ross', 'Rachel', 'Phoebe', 'Monica', 'Chandler') or \
        cur_video_emotion not in ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'):
        return 'Other', 'Other', []

    num_frames = 8
    # Initialize an EncodedVideo helper class and load the video
    video = cv2.VideoCapture(os.path.join(data_path, folder_name,
                                          video_name))  #读入视频文件
    fps = int(video.get(7))
    if fps == 'Not known':
        fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return 'Other', 'Other', []
    # print(video_name)
    interval = fps // num_frames
    clip_num = []
    for i in range(num_frames):
        clip_num.append((i + 1) * interval)
    rval = video.isOpened()  # 判断视频是否打开 返回True或Flase
    # frames = []
    filenames = []
    c = 1
    while rval:  # 循环读取视频帧
        rval, frame = video.read()
        if rval:
            if c in clip_num:
                # tensor保存进csv
                savename = video_name[:-4] + '-' + str(
                    clip_num.index(c)) + '.jpg'
                cv2.imwrite(
                    os.path.join(data_path, save_folder_name, savename), frame)
                filenames.append(savename)
                # img = trans(frame)
                # frames += [img.numpy()]
            c = c + 1
    video.release()
    if len(filenames) == num_frames:
        return cur_video_speaker, cur_video_emotion, filenames
    else:
        return 'Other', 'Other', filenames
    # transformed_video_tensor = torch.Tensor(frames)
    # return transformed_video_tensor, cur_video_speaker, cur_video_emotion, fps


def main():
    # # ------------------ transforms ------------------

    # ------------------ file ------------------
    train_data_path = '/workspace/chi149/MELD/MELD.Raw/train'
    train_folder_name = 'train_splits'
    train_info_csv_name = 'train_sent_emo.csv'
    train_save_folder_name = 'train_video_pic8'

    test_data_path = '/workspace/chi149/MELD/MELD.Raw/test'
    test_folder_name = 'output_repeated_splits_test'
    test_info_csv_name = 'test_sent_emo.csv'
    test_save_folder_name = 'test_video_pic8'

    print("------------------ train pickle ------------------")
    # train_df = get_df(train_data_path, train_folder_name, train_info_csv_name,
    #                   trans_train)
    # train_df.to_pickle(os.path.join(train_data_path, 'train_video_tensor.pkl'))
    train_video_names = []
    train_video_speaker = {}
    train_video_emotion = {}
    train_video_pic_names = {}

    train_video_list = os.listdir(
        os.path.join(train_data_path, train_folder_name))
    train_data_info = pd.read_csv(
        os.path.join(train_data_path, train_info_csv_name))
    for video_name in tqdm(train_video_list,
                           desc='getting video tensor pickle',
                           ncols=100):
        speaker, emotion, filenames = getVideoTensor(train_data_path,
                                                     train_folder_name,
                                                     video_name,
                                                     train_save_folder_name,
                                                     train_data_info)
        if emotion != 'Other' and speaker != 'Other' and len(filenames) == 8:
            train_video_names.append(video_name)
            train_video_speaker[video_name] = speaker
            train_video_emotion[video_name] = emotion
            train_video_pic_names[video_name] = filenames
        else:
            pass
    print(len(train_video_names))

    train_video_speaker_save = "/workspace/chi149/MELD/MELD.Raw/train/train_video_speaker_dic_8.pkl"
    train_video_emotion_save = "/workspace/chi149/MELD/MELD.Raw/train/train_video_emotion_dic_8.pkl"
    train_video_filename_save = "/workspace/chi149/MELD/MELD.Raw/train/train_video_filename_dic_8.pkl"
    with open(train_video_speaker_save, 'wb') as f:
        pickle.dump(train_video_speaker, f)
    print("train video speaker dic save pickle success")
    with open(train_video_emotion_save, 'wb') as f:
        pickle.dump(train_video_emotion, f)
    print("train video emotion dic save pickle success")
    with open(train_video_filename_save, 'wb') as f:
        pickle.dump(train_video_pic_names, f)
    print("train video filename dic save pickle success")

    print("------------------ test pickle ------------------")
    # test_df = get_df(test_data_path, test_folder_name, test_info_csv_name,
    #                  trans_test)
    # test_df.to_pickle(os.path.join(test_data_path, 'test_video_tensor.pkl'))
    test_video_names = []
    test_video_speaker = {}
    test_video_emotion = {}
    test_video_pic_names = {}

    test_video_list = os.listdir(os.path.join(test_data_path,
                                              test_folder_name))
    test_data_info = pd.read_csv(
        os.path.join(test_data_path, test_info_csv_name))
    for video_name in tqdm(test_video_list,
                           desc='getting video tensor pickle',
                           ncols=100):
        speaker, emotion, filenames = getVideoTensor(test_data_path,
                                                     test_folder_name,
                                                     video_name,
                                                     test_save_folder_name,
                                                     test_data_info)
        if emotion != 'Other' and speaker != 'Other' and len(filenames) == 8:
            test_video_names.append(video_name)
            test_video_speaker[video_name] = speaker
            test_video_emotion[video_name] = emotion
            test_video_pic_names[video_name] = filenames
        else:
            pass
    print(len(test_video_names))

    test_video_speaker_save = "/workspace/chi149/MELD/MELD.Raw/test/test_video_speaker_dic_8.pkl"
    test_video_emotion_save = "/workspace/chi149/MELD/MELD.Raw/test/test_video_emotion_dic_8.pkl"
    test_video_filename_save = "/workspace/chi149/MELD/MELD.Raw/test/test_video_filename_dic_8.pkl"
    with open(test_video_speaker_save, 'wb') as f:
        pickle.dump(test_video_speaker, f)
    print("test video speaker dic save pickle success")
    with open(test_video_emotion_save, 'wb') as f:
        pickle.dump(test_video_emotion, f)
    print("test video emotion dic save pickle success")
    with open(test_video_filename_save, 'wb') as f:
        pickle.dump(test_video_pic_names, f)
    print("test video filename dic save pickle success")


if __name__ == '__main__':
    main()