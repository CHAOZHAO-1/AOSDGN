#author:zhaochao time:2021/10/18
import pickle
import libmr
import  numpy as np

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial.distance import mahalanobis

def maha_distance(p,cen,VI):


    zero_f = p - cen

    dis=np.sqrt(np.dot(np.dot(zero_f, VI), zero_f.T))

    return dis



def mahalanobis_distance(p, distr):

    cov = np.cov(distr, rowvar=False)


    cov=np.linalg.inv(cov)

    avg_distri = np.average(distr, axis=0)# 选取分布中各维度均值所在点


    dis = mahalanobis(p, avg_distri, cov)
    return dis


def extra_data(sour,tar,task,way1,dataset_class):

    sour1 = sour[0]
    sour2 = sour[1]
    sour3 = sour[2]

    tar = tar
    task = task
    way1 = way1

    sour_name1 = 'EVTvis/'+str(sour1) +str(sour2)+str(sour3)+ str(tar) + way1 + task + str(sour1)
    fea1 = pickle.load(open(sour_name1 + '.pkl', 'rb'))
    sour_input1 = fea1['feature']

    sour_name2 = 'EVTvis/' + str(sour1) + str(sour2) + str(sour3) + str(tar) + way1 + task + str(sour2)
    fea2 = pickle.load(open(sour_name2 + '.pkl', 'rb'))
    sour_input2 = fea2['feature']

    sour_name3 = 'EVTvis/' + str(sour1) + str(sour2) + str(sour3) + str(tar) + way1 + task + str(sour3)
    fea3 = pickle.load(open(sour_name3 + '.pkl', 'rb'))
    sour_input3 = fea3['feature']

    _,length=sour_input1.shape

    sour_input=np.zeros((dataset_class*800*3,length))

    for indexclass in range (1,dataset_class+1):

        sour_input[2400*(indexclass-1):2400*(indexclass-1)+800]=sour_input1[800*(indexclass-1):800*indexclass]
        sour_input[2400*(indexclass-1)+800:2400*(indexclass-1)+1600]=sour_input2[800*(indexclass-1):800*indexclass]
        sour_input[2400*(indexclass-1)+1600:2400*(indexclass-1)+2400]=sour_input3[800*(indexclass-1):800*indexclass]


    tar_name = 'EVTvis/'+str(sour1) + str(sour2) + str(sour3) + str(tar) + way1  +  task  + str(tar)
    fea = pickle.load(open(tar_name + '.pkl', 'rb'))
    tar_input = fea['feature']

    T_predict = fea['predict_label'].reshape(-1, 1)

    return sour_input, tar_input, T_predict



def fitM(sour_input, tar_input, T_predict,class_num,partial,threshold):

    all_category = np.linspace(1, class_num, class_num, endpoint=True)

    notargetclasses = list(set(all_category) ^ set(partial))

    source_num=len(partial)

    dis_mean=np.zeros((source_num,256))

    dis_intev_cov=np.zeros((source_num,256,256))

    testLabel = np.zeros(200 * class_num) + len(partial)

    kk = 1

    for retain_class in partial:

        sour = sour_input[2400 * (retain_class - 1):2400 * retain_class]
        #
        center = np.sum(sour, axis=0) / 2400

        S = np.cov(sour,rowvar=False)

        S_s=np.identity(256)*0.00000000001


        AI = np.linalg.inv(S+S_s)

        dis_mean[kk-1]=center

        dis_intev_cov[kk-1]=AI

        testLabel[200 * (retain_class - 1):200 * retain_class] = kk - 1

        kk = kk + 1


    test_score=np.zeros(200*class_num)




    for j in range(200*class_num):


        M_dis = np.zeros(source_num)

        for mmm in range(source_num):

            zero_f =  tar_input[j]-dis_mean[mmm]

            indi_dis=np.sqrt(np.dot(np.dot(zero_f,dis_intev_cov[mmm]),zero_f.T))

            # indi_dis=mahalanobis_distance(tar_input[j],sour)

            M_dis[mmm]=indi_dis

        score =M_dis.min()

        test_score[j]=score

        if score > threshold:

            T_predict[j] = len(partial)


    known_dis=0
    unknown_dis=0


    for sss in partial:

        known_dis+=test_score[(sss-1)*200:sss*200].mean()

    print('known_dis:{}'.format(known_dis/len(partial)))



    for ddd in notargetclasses:

        unknown_dis+=test_score[int(ddd-1)*200:int(ddd)*200].mean()

    print('unknown_dis:{}'.format(unknown_dis / len(notargetclasses)))




    # print(test_score[:200].mean())
    # print(test_score[200:400].mean())
    # print(test_score[400:600].mean())
    # print(test_score[600:800].mean())


    known_label = np.zeros(200 * len(partial))
    for i in range(200 * len(partial)):
        known_label[i] = i // 200

    unknown_label = np.zeros(200 * (class_num - len(partial))) + len(partial)

    long = len(partial)
    if long != 0:
        b = np.linspace(0, 199, 200).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (partial[i] - 1) * 200 + b)

        T_predict_unkown = np.delete(T_predict, removelist, axis=0)


    long = len(notargetclasses)
    if long != 0:
        b = np.linspace(0, 199, 200).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetclasses[i] - 1) * 200 + b)
        T_predict_kown = np.delete(T_predict, removelist, axis=0)

        OS_ = accuracy_score(known_label, T_predict_kown)

        UNK = accuracy_score(unknown_label, T_predict_unkown)

        OS = (len(partial) * OS_ + UNK) / (len(partial) + 1)

        print('Overll_acc:{}; known acc:{}; unkown acc:{}'.format(OS, OS_, UNK))


    if long==0:

        T_predict_kown=T_predict

        OS_ = accuracy_score(known_label, T_predict_kown)

        UNK = 0

        OS = OS_

        print('Overll_acc:{}; known acc:{}; unkown acc:{}'.format(OS, OS_, UNK))

    return OS, OS_, UNK



