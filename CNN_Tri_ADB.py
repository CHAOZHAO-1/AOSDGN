#author:zhaochao time:2021/10/26
#author:zhaochao time:2021/10/26

import pickle

import torch
import torch.nn.functional as F

from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
import torch.nn as nn
import  time
import numpy as np
import  random
from utils import *

from Mahalanobis import *

from ext import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings

momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4


def calculate_center (feature,label,src_class):

    n, d = feature.shape

    # get labels
    s_labels= label  # 得到源域和目标域标签

    # image number in each class
    ones = torch.ones_like(s_labels, dtype=torch.float)
    zeros = torch.zeros(src_class)

    zeros = zeros.cuda()
    s_n_classes = zeros.scatter_add(0, s_labels, ones)

    # image number cannot be 0, when calculating centroids
    ones = torch.ones_like(s_n_classes)
    s_n_classes = torch.max(s_n_classes, ones)

    # calculating centroids, sum and divide
    zeros = torch.zeros(src_class, d)

    zeros = zeros.cuda()
    s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), feature)

    current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(src_class, 1))

    return current_s_centroid







def cal_score(feature, center):

    size,d=feature.shape

    distance=torch.zeros((size,len(partial)))

    dis_Loss=nn.MSELoss(reduction='none')

    for k in range(len(partial)):

        br_center=center[k].reshape(1,-1).repeat(size,1)

        distance[:,k]=torch.sum(dis_Loss(feature,br_center),dim=1)/d

        score,_=torch.min(distance,dim=1)

    return score







def train(model):
    src_iter = iter(src_loader)


    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            {'params':criterion_boundary.parameters(),'lr':10*LEARNING_RATE}
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()



        src_label=src_label[:,0]

        optimizer.zero_grad()

        src_pred, src_feature= model(src_data)

        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)

        class_center=calculate_center(src_feature,src_label,len(partial))


        loss_adb, delta = criterion_boundary(src_feature, class_center, src_label)

        print(delta)

        selector = BatchHardTripletSelector()
        anchor, pos, neg = selector(src_feature, src_label)
        triplet_loss = TripletLoss(margin=1).cuda()


        triplet = triplet_loss(anchor, pos, neg)


        loss = cls_loss+loss_adb+triplet

        loss.backward()

        optimizer.step()

        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tADB_Loss: {:.6f}\tTri_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(),loss_adb,triplet))

        if i % (log_interval * 10) == 0:


            train_correct, train_loss = test_source(model, src_loader)

            tar_OS, tar_OS_, tar_UNK = test_target(model, tgt_test_known_loader, tgt_test_unknown_loader,class_center,delta)






def test_source(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label=tgt_test_label[:,0]
            # print(tgt_test_data)
            tgt_pred,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name,test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


def test_target(model,test_know_loader,test_unknow_loader,class_center,delta):
    model.eval()
    test_loss = 0
    correct_know = 0
    correct_unknow = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_know_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred,tgt_feature= model(tgt_test_data)

            score = cal_score(tgt_feature, class_center)


            pred =open_classify(tgt_feature,class_center,delta,len(partial),th)


            correct_know += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

        for tgt_test_data, tgt_test_label in test_unknow_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred,tgt_feature= model(tgt_test_data)

            score = cal_score(tgt_feature, class_center)



            pred = open_classify(tgt_feature, class_center, delta, len(partial),th)


            correct_unknow += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    OS_=np.array(correct_know)/len(test_know_loader.dataset)
    UNK=np.array(correct_unknow)/len(test_unknow_loader.dataset)

    if len(partial) == class_num:
        OS = OS_
    else:
        OS = (len(Source_class[taskindex]) * OS_ + UNK) / (len(Source_class[taskindex]) + 1)

    HSCORE = 2 * OS * UNK / (OS + UNK)

    print(
        '\n - {} set, hscore:{} ,global Accuracy:({:.2f}%), know Accuracy: {}/{} ({:.2f}%),unknow Accuracy:{}/{}  ({:.2f}%)\n'.format
        (tgt_name, HSCORE * 100, OS * 100, correct_know, len(test_know_loader.dataset), OS_ * 100, correct_unknow,
         len(test_unknow_loader.dataset), UNK * 100))


    return OS,OS_,UNK


def test_target_pre(model,test_loader,class_center,delta):
    model.eval()
    test_loss = 0
    correct_know = 0
    correct_unknow = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred,tgt_feature= model(tgt_test_data)

            pred =open_classify(tgt_feature,class_center,delta,len(partial),th)

            pred = pred.data.cpu().numpy()



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)
    iteration = 10000
    batch_size = 1024
    lr = 0.001


    th=1.3

    dataset = 'LW'

    FFT = True

    class_num = 4

    Task_name = np.array(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10',
                          ])
    src_tar = np.array([[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5],
                        [2, 3, 5, 4], [2, 3, 5, 4], [2, 3, 5, 4], [2, 3, 5, 4], [2, 3, 5, 4], [2, 3, 5, 4],

                        ])

    Source_class = np.array([
        [1, 2, 3],
        [1, 3, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 2, 3],
        [1, 3, 4],
        [1, 2],
        [1, 3],
        [1, 4],

    ])


    for taskindex in range(10):
        source1 = src_tar[taskindex][0]
        source2 = src_tar[taskindex][1]
        source3 = src_tar[taskindex][2]
        target = src_tar[taskindex][3]
        src = src_tar[taskindex][:-1]

        all_category = np.linspace(1, class_num, class_num, endpoint=True)
        partial = Source_class[taskindex]
        notargetclasses = list(set(all_category) ^ set(partial))  # 要删去的类别

        for repeat in range(1):
            root_path = '/home/dlzhaochao/deeplearning/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'
            src_name1 = 'load' + str(source1) + '_train'
            src_name2 = 'load' + str(source2) + '_train'
            src_name3 = 'load' + str(source3) + '_train'

            tgt_name = 'load' + str(target) + '_train'
            test_name = 'load' + str(target) + '_test'

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training(notargetclasses, root_path, src_name1, src_name2, src_name3,
                                                      src, FFT, len(Source_class[taskindex]),
                                                      batch_size, kwargs)

            tgt_test_known_loader = data_loader_1d.load_testing_known(notargetclasses, root_path, test_name, FFT,
                                                                      len(Source_class[taskindex]),
                                                                      batch_size, kwargs)

            tgt_test_unknown_loader = data_loader_1d.load_testing_unknown(partial, root_path, test_name, FFT,
                                                                          len(notargetclasses), batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)
            model = models.CNN_1D(num_classes=len(partial))

            criterion_boundary = BoundaryLoss(num_labels=len(partial), feat_dim=256)

            # get_parameter_number(model) 计算模型训练参数个数
            print(model)
            if cuda:
                model.cuda()
            train(model)






































