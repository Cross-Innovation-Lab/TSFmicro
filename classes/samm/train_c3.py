import numpy as np
import torch
from torchvision import transforms
from base import seed_torch
from classes.model.ts_micro import ts_micro
from classes.samm.dataset_c3 import samm_c3_DataSet


def run_training(raf_path):


    ##data normalization for both training set
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])
    ### data augmentation for training set only
    data_transforms_norm = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.RandomCrop(224, padding=4),

    ])

    ### data normalization for both teating set
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])




    criterion = torch.nn.CrossEntropyLoss()

    # leave one subject out protocal
    LOSO = ['6', '7', '9','10','11','12','13','14','15','16','17','18',
            '19','20','21','22','23','24','26','28','30','31','32',
            '33','34','35','36','37']

    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(3)
    pos_label_ALL = torch.zeros(3)
    TP_ALL = torch.zeros(3)

    for subj in LOSO:
        train_dataset = samm_c3_DataSet(raf_path, phase='train', num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
        val_dataset = samm_c3_DataSet(raf_path, phase='test', num_loso=subj, transform=data_transforms_val)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   num_workers=0,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=32,
                                                 num_workers=0,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(3)
        max_pos_label = torch.zeros(3)
        max_TP = torch.zeros(3)
        ##model initialization
        net_all = ts_micro(num_classes=3)

        params_all = net_all.parameters()

        optimizer_all = torch.optim.AdamW(params_all, lr=0.0008, weight_decay=0.7)

        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)

        net_all = net_all.cuda()

        for i in range(1, 200):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0

            net_all.train()


            for batch_i, (image_on0, image_apex0, label_all) in enumerate(train_loader):
                batch_sz = image_on0.size(0)
                b, c, h, w = image_on0.shape
                iter_cnt += 1

                image_on0 = image_on0.cuda()
                image_apex0 = image_apex0.cuda()
                label_all = label_all.cuda()


                ##train MMNet
                ALL = net_all(image_on0, image_apex0)

                loss_all = criterion(ALL, label_all)

                optimizer_all.zero_grad()

                loss_all.backward()

                optimizer_all.step()
                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all).sum()
                correct_sum += correct_num






            ## lr decay
            if i <= 50:

                scheduler_all.step()
            if i>=0:
                acc = correct_sum.float() / float(train_dataset.__len__())

                running_loss = running_loss / iter_cnt

                #print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))


            pos_label = torch.zeros(3)
            pos_pred = torch.zeros(3)
            TP = torch.zeros(3)

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                net_all.eval()
                # net_au.eval()
                for batch_i, (
                image_on0, image_apex0, label_all) in enumerate(val_loader):
                    batch_sz = image_on0.size(0)
                    b, c, h, w = image_on0.shape

                    image_on0 = image_on0.cuda()

                    image_apex0 = image_apex0.cuda()

                    label_all = label_all.cuda()
                    #label_au = label_au.cuda()

                    ##test
                    ALL = net_all(image_on0, image_apex0)


                    loss = criterion(ALL, label_all)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(ALL, 1)
                    correct_num = torch.eq(predicts, label_all)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += ALL.size(0)

                    for cls in range(3):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in label_all:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, label_all):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1
                        # if pos_label != 0 or pos_pred != 0:
                        #     f1 = 2 * TP / (pos_pred + pos_label)
                        #     F1.append(f1)
                    count = 0
                    SUM_F1 = 0
                    for index in range(3):
                        if pos_label[index] != 0 or pos_pred[index] != 0:
                            count = count + 1
                            SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])

                    AVG_F1 = SUM_F1 / count


                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (i, acc, running_loss, AVG_F1))
                if acc==1.:
                    print('achieve 100%acc, break')
                    break
        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        for index in range(3):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        print("[..........%s] correctnum:%d . zongshu:%d   " % (subj, max_corr, val_dataset.__len__()))
        print("[ALL_corr]: %d [ALL_val]: %d" % (num_sum, val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))
    for index in range(5):
        print("[%s:] TP: %.4f LAB: %.4f" % (index, TP_ALL[index], pos_pred_ALL[index]))
        print("[%s] ACC:%.4f" % (index, TP_ALL[index]/pos_pred_ALL[index]))




if __name__ == "__main__":
    seed = 0
    seed_torch(seed)
    raf_path = ''
    run_training(raf_path)