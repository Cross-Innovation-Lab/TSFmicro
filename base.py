import os.path
import random
import sys

import cv2
import numpy as np
import torch

def flow_difference(onset, apex, offset, file_path=None, sub=None, f=None):
    onset = onset
    apex = apex
    offset = offset
    diff_dict = {}
    ran = int(0.15 * (apex - onset) / 4)
    ran_off = int(0.15 * (offset - apex) / 4)
    #ran = min(ran, 3)
    path = os.path.join(file_path, 'Cropped/', 'sub%02d' % int(sub), f)
    onset_path = path + '/reg_img' + str(onset) + '.jpg'
    apex_path = path + '/reg_img' + str(apex) + '.jpg'
    onset_img = cv2.imread(onset_path)
    apex_img = cv2.imread(apex_path)
    stand = np.linalg.norm(apex_img - onset_img)
    diff_dict[(onset, apex)] = stand
    for index1 in range(1, ran+1):
        onset0 = onset + index1
        onset0_path = path + '/reg_img' + str(onset0) + '.jpg'
        onset0_img = cv2.imread(onset0_path)
        diff0 = np.linalg.norm(apex_img - onset0_img)
        diff_dict[(onset0, apex)] = diff0
        for index2 in range(1, ran+1):
            apex0 = apex - index2
            apex0_path = path + '/reg_img' + str(apex0) + '.jpg'
            apex0_img = cv2.imread(apex0_path)
            diff2 = np.linalg.norm(apex0_img - onset0_img)
            diff_dict[(onset0, apex0)] = diff2
    for index3 in range(1, ran+1):
        apex1 = apex - index3
        apex1_path = path + '/reg_img' + str(apex1) + '.jpg'
        apex1_img = cv2.imread(apex1_path)
        diff3 = np.linalg.norm(apex1_img - onset_img)
        diff_dict[(onset, apex1)] = diff3
    # for index4 in range(1, ran_off+1):
    #     apex2 = apex + index4
    #     apex2_path = path + '/reg_img' + str(apex2) + '.jpg'
    #     apex2_img = cv2.imread(apex2_path)
    #     diff4 = np.linalg.norm(apex2_img - onset_img)
    #     diff_dict[(onset, apex2)] = diff4
    #     for index5 in range(1, ran_off+1):
    #         onset5 = onset + index5
    #         onset5_path = path + '/reg_img' + str(onset5) + '.jpg'
    #         onset5_img = cv2.imread(onset5_path)
    #         diff5 = np.linalg.norm(apex2_img - onset5_img)
    #         diff_dict[(onset5, apex2)] = diff5

    max_diff_key = max(diff_dict, key=diff_dict.get)

    onset_index, apex_index = max_diff_key

    return onset_index, apex_index

def samm_difference(onset, apex, offset, file_path=None, sub=None, f=None):
    onset = onset
    apex = apex
    offset = offset
    diff_dict = {}
    ran = int(0.1 * (apex - onset))
    ran_off = int(0.1 * (offset - apex))
    #ran = min(ran, 3)
    path = os.path.join(file_path, '%03d' % int(sub), f)
    onset_path = path + '/' + '%03d' % int(sub) + '_' + str(onset) + '.jpg'
    apex_path = path + '/' + '%03d' % int(sub) + '_' + str(apex) + '.jpg'
    onset_img = cv2.imread(onset_path)
    apex_img = cv2.imread(apex_path)
    stand = np.linalg.norm(apex_img - onset_img)
    diff_dict[(onset, apex)] = stand
    for index1 in range(1, ran+1):
        onset0 = onset + index1
        onset0_path = path + '/' + '%03d' % int(sub) + '_' + str(onset0) + '.jpg'
        onset0_img = cv2.imread(onset0_path)
        diff0 = np.linalg.norm(apex_img - onset0_img)
        diff_dict[(onset0, apex)] = diff0
        for index2 in range(1, ran+1):
            apex0 = apex - index2
            apex0_path = path + '/' + '%03d' % int(sub) + '_' + str(apex0) + '.jpg'
            apex0_img = cv2.imread(apex0_path)
            diff2 = np.linalg.norm(apex0_img - onset0_img)
            diff_dict[(onset0, apex0)] = diff2
    for index3 in range(1, ran+1):
        apex1 = apex - index3
        apex1_path = path + '/' + '%03d' % int(sub) + '_' + str(apex1) + '.jpg'
        apex1_img = cv2.imread(apex1_path)
        diff3 = np.linalg.norm(apex1_img - onset_img)
        diff_dict[(onset, apex1)] = diff3
    # for index4 in range(1, ran_off+1):
    #     apex2 = apex + index4
    #     apex2_path = path + '/reg_img' + str(apex2) + '.jpg'
    #     apex2_img = cv2.imread(apex2_path)
    #     diff4 = np.linalg.norm(apex2_img - onset_img)
    #     diff_dict[(onset, apex2)] = diff4
    #     for index5 in range(1, ran_off+1):
    #         onset5 = onset + index5
    #         onset5_path = path + '/reg_img' + str(onset5) + '.jpg'
    #         onset5_img = cv2.imread(onset5_path)
    #         diff5 = np.linalg.norm(apex2_img - onset5_img)
    #         diff_dict[(onset5, apex2)] = diff5

    max_diff_key = max(diff_dict, key=diff_dict.get)

    onset_index, apex_index = max_diff_key

    return onset_index, apex_index


def casme3_difference(onset, apex, offset, file_path=None, sub=None, f=None):
    onset = onset
    apex = apex
    offset = offset
    diff_dict = {}
    ran = int(0.15 * (apex - onset) / 4)
    path = os.path.join(file_path, sub, f)
    onset_path = path + '/' + str(onset) + '.jpg'
    apex_path = path + '/' + str(apex) + '.jpg'
    onset_img = cv2.imread(onset_path)
    apex_img = cv2.imread(apex_path)
    stand = np.linalg.norm(apex_img - onset_img)
    diff_dict[(onset, apex)] = stand
    for index1 in range(1, ran+1):
        onset0 = onset + index1
        onset0_path = path + '/' + str(onset0) + '.jpg'
        onset0_img = cv2.imread(onset0_path)
        diff0 = np.linalg.norm(apex_img - onset0_img)
        diff_dict[(onset0, apex)] = diff0
        for index2 in range(1, ran+1):
            apex0 = apex - index2
            apex0_path = path + '/' + str(apex0) + '.jpg'
            apex0_img = cv2.imread(apex0_path)
            diff2 = np.linalg.norm(apex0_img - onset0_img)
            diff_dict[(onset0, apex0)] = diff2
    for index3 in range(1, ran+1):
        apex1 = apex - index3
        apex1_path = path + '/' + str(apex1) + '.jpg'
        apex1_img = cv2.imread(apex1_path)
        diff3 = np.linalg.norm(apex1_img - onset_img)
        diff_dict[(onset, apex1)] = diff3
    max_diff_key = max(diff_dict, key=diff_dict.get)
    onset_index, apex_index = max_diff_key
    return onset_index, apex_index

def seed_torch(seed=2):
    print('seed=',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

class Logger(object):
    def __init__(self,log_name):
        self.terminal = sys.stdout
        self.log_name=log_name
        self.log = open(self.log_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass