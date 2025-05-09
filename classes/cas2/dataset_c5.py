import os

import cv2
import pandas as pd
import torch
import torch.utils.data as data
from base import flow_difference


class cas2_c5_DataSet(data.Dataset):
    def __init__(self, raf_path, phase,num_loso, transform = None, basic_aug = False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        SUBJECT_COLUMN =0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6


        df = pd.read_excel(os.path.join(self.raf_path, 'CASME2-coding-20190701.xlsx'),usecols=[0,1,3,4,5,7,8])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject']!=num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names =[]
        a=0
        b=0
        c=0
        d=0
        e=0
        # use aligned images for training/testing
        for (f,sub,onset,apex,offset,label_all,label_au) in zip(File_names,Subject,Onset_num,Apex_num,Offset_num,Label_all,Label_au):


            if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':

                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.sub.append(sub)
                self.file_names.append(f)
                if label_all == 'happiness':
                    self.label_all.append(0)
                    a=a+1
                elif label_all == 'repression':
                    self.label_all.append(1)
                    b=b+1
                elif label_all == 'disgust':
                    self.label_all.append(2)
                    c=c+1
                elif label_all == 'surprise':
                    self.label_all.append(3)
                    d=d+1
                else:
                    self.label_all.append(4)
                    e=e+1

            # label_au =label_au.split("+")
                if isinstance(label_au, int):
                    self.label_au.append([label_au])
                else:
                    label_au = label_au.split("+")
                    self.label_au.append(label_au)






            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])

            on0, apex0 = flow_difference(int(onset), int(apex), int(offset), self.raf_path, sub, f)
            on0 = str(on0)
            apex0 = str(apex0)
        else:##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]

            on0 = str(onset)
            apex0 = str(apex)


            sub = str(self.sub[idx])
            f = str(self.file_names[idx])


        on0 ='reg_img'+on0+'.jpg'

        apex0 ='reg_img' + apex0 + '.jpg'

        path_on0 = os.path.join(self.raf_path, 'Cropped/', 'sub%02d' % int(sub), f, on0)

        path_apex0 = os.path.join(self.raf_path, 'Cropped/', 'sub%02d' % int(sub), f, apex0)

        image_on0 = cv2.imread(path_on0)

        image_apex0 = cv2.imread(path_apex0)


        image_on0 = image_on0[:, :, ::-1] # BGR to RGB

        image_apex0 = image_apex0[:, :, ::-1]

        label_all = self.label_all[idx]
        label_au = self.label_au[idx]

        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)

            image_apex0 = self.transform(image_apex0)
            ALL = torch.cat(
                (image_on0, image_apex0), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]

            image_apex0 = ALL[3:6, :, :]


            return image_on0, image_apex0, label_all
