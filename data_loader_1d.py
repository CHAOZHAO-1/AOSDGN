from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio
import math

def wgn(x, snr):
    Ps = np.sum(abs(x)**2,axis=1)/len(x)
    Pn = Ps/(10**((snr/10)))
    row,columns=x.shape
    Pn = np.repeat(Pn.reshape(-1,1),columns, axis=1)

    noise = np.random.randn(row,columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

def zscore(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Z = (Z - Zmin.reshape(-1,1)) / (Zmax.reshape(-1,1) - Zmin.reshape(-1,1))
    return Z


def min_max(Z):
    Zmin = Z.min(axis=1)

    Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
    return Z



def load_training(notargetlist,root_path,dir1,dir2,dir3,src_list, fft1, class_num , batch_size, kwargs):
    src_list=torch.from_numpy(src_list)

    data = scio.loadmat(root_path)
    if fft1==True:
        train_fea_1=zscore((min_max(abs(fft(data[dir1]))[:, 0:512])))
    if fft1==False:
        train_fea_1 = zscore(data[dir1])

    long = len(notargetlist)
    if long != 0:
        b = np.linspace(0, 799, 800).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 800 + b)
        train_fea_1 = np.delete(train_fea_1, removelist, axis=0)


    train_label_1 = torch.zeros((800 * class_num,2))
    for i in range(800 * class_num):
        train_label_1[i][0] = i // 800

        train_label_1[i][1] =0


    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea_2= zscore((min_max(abs(fft(data[dir2]))[:, 0:512])))
    if fft1 == False:
        train_fea_2= zscore(data[dir2])

    long = len(notargetlist)
    if long != 0:
        b = np.linspace(0, 799, 800).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 800 + b)
        train_fea_2 = np.delete(train_fea_2, removelist, axis=0)


    train_label_2= torch.zeros((800 * class_num, 2))
    for i in range(800 * class_num):
        train_label_2[i][0] = i // 800

        train_label_2[i][1] = 1


    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea_3= zscore((min_max(abs(fft(data[dir3]))[:, 0:512])))
    if fft1 == False:
        train_fea_3= zscore(data[dir3])

    long = len(notargetlist)
    if long != 0:
        b = np.linspace(0, 799, 800).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 800 + b)
        train_fea_3 = np.delete(train_fea_3, removelist, axis=0)


    train_label_3= torch.zeros((800 * class_num, 2))
    for i in range(800 * class_num):
        train_label_3[i][0] = i // 800

        train_label_3[i][1] =2


    train_fea=np.vstack((train_fea_1,train_fea_2,train_fea_3))

    train_label=torch.cat([train_label_1,train_label_2,train_label_3], dim=0)


    train_label = train_label.long()
    train_fea=torch.from_numpy(train_fea)
    train_fea= torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, dir, fft1, class_num, batch_size, kwargs):
    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea = zscore((min_max(abs(fft(data[dir]))[:, 0:512])))
    if fft1 == False:
        train_fea = zscore(data[dir])

    train_label = torch.zeros((200 * class_num))
    for i in range(200 * class_num):
        train_label[i] = i // 200

    print(train_fea.shape)
    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea)
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    return train_loader



def load_testing_known(notargetlist, root_path, dir, fft1, class_num, batch_size, kwargs):
    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea = zscore((min_max(abs(fft(data[dir]))[:, 0:512])))
    if fft1 == False:
        train_fea = zscore(data[dir])
    train_label = torch.zeros((200 * class_num))
    for i in range(200 * class_num):
        train_label[i] = i // 200
    #
    long = len(notargetlist)
    if long!=0:
        b = np.linspace(0, 199, 200).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 200 + b)
        train_fea = np.delete(train_fea, removelist, axis=0)


    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea)
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader



def load_testing_unknown(notargetlist, root_path, dir, fft1, class_num, batch_size, kwargs):
    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea = zscore((min_max(abs(fft(data[dir]))[:, 0:512])))
    if fft1 == False:
        train_fea = zscore(data[dir])
    train_label = torch.zeros((200 * class_num))

    for i in range(200 * class_num):
        train_label[i] = len(notargetlist)

    #
    long = len(notargetlist)
    if long!=0:
        b = np.linspace(0, 199, 200).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 200 + b)
        train_fea = np.delete(train_fea, removelist, axis=0)


    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea)
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader








