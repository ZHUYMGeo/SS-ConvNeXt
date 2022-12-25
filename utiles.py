import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import random
from datetime import datetime
import pdb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import sys
from random import randint
from sklearn.decomposition import PCA
import time

plt.style.use(['ieee','no-latex'])


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def show_curve(val_acc, train_loss):
    with plt.style.context(['ieee', 'no-latex']):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot(train_loss, color=color)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color2 = 'tab:blue'
        ax2.plot(val_acc, color=color2)
        ax2.set_ylabel('Accuracy', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.tight_layout()
        plt.show()


def read_data(data_filepath, target_filepath):
    data_init = sio.loadmat(data_filepath)
    data = list(filter(lambda x: isinstance(x, np.ndarray), data_init.values()))[0]
    data = data.transpose(2, 0, 1)
    data = normalize(data)
    target_init = sio.loadmat(target_filepath)
    target = list(filter(lambda x: isinstance(x, np.ndarray), target_init.values()))[0]

    return data, target
           # cat_train_data, cat_train_label, cat_val_data, cat_val_label


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise


def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    data = data.transpose(2, 0, 1)
    return data


def normalize(data):
    data = data.astype(np.float32)
    for i in range(len(data)):
        ma = np.max(data[i, :, :])
        mi = np.min(data[i, :, :])
        data[i, :, :] = (data[i, :, :] - mi) / (ma - mi)

    return data


def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


def get_mask(target, train_prop, val_prop, save_dir=None):
    assert train_prop+val_prop <= 1
    train_mask = np.zeros([target.shape[0], target.shape[1]])
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, int(target.max())+1):
        indx = np.argwhere(target == i)
        train_num = max(round(train_prop*len(indx)), 3)
        val_num = max(round(val_prop*len(indx)), 3)

        np.random.shuffle(indx)
        train_indx = indx[: train_num]
        val_indx = indx[train_num:train_num+val_num]
        test_indx = indx[train_num+val_num:]

        train_mask[train_indx[:, 0], train_indx[:, 1]] = 1
        val_mask[val_indx[:, 0], val_indx[:, 1]] = 1
        test_mask[test_indx[:, 0], test_indx[:, 1]] = 1

    if save_dir:
        folder_name = 'masks_{}_{}'.format(train_prop, val_prop)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'),
                    {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'),
                    {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'),
                    {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


def get_each_class_num(target, total_num, min_num=3):  # 固定总的训练样本

    class_num = target.max().astype(int)  # 类别的数量
    assert total_num >= (class_num * min_num)   # 每类至少3个样本

    target_count = target[target != 0]  # 所有有标签的样本个数  去除背景
    num = float(len(target_count))
    num_i_ = pd.value_counts(target_count).sort_index().values  # 每一类的总样本数
    num_i = np.floor(num_i_ / num * total_num).astype(np.int)
    num_i[num_i < min_num] = min_num

    if num_i.sum() > total_num:
        i = 0
        max_idx = (-num_i).argsort()  # 从大到小排序后，返回对应的数组的索引。[0 - 15]
        while num_i.sum() != total_num:
            num_i[max_idx[i]] = max(min_num, num_i[max_idx[i]] - 1)
            i = i + 1 if (i + 1) < len(num_i) else 0

    if num_i.sum() < total_num:
        i = 0
        max_idx = num_i.argsort()   # 从小到大排序后，返回对应的数组的索引。[0 - 15]
        while num_i.sum() != total_num:
            num_i[max_idx[i]] = min(num_i_[max_idx[i]], num_i[max_idx[i]] + 1)
            i = i + 1 if (i + 1) < len(num_i) else 0

    return num_i


def get_fixed_number_masks(target, train_num, val_num, save_dir=None):
    """get masks that be used to extracted training/val/test samples,
    training samples number is determined by a fixed number

    Parameters
    ----------
    target: ndarray, H*W, the ground truth of HSI
    train_num: int, the number of training samples, at least 50  16*3=48
    val_num: float, the proportion of validation samples, e.g. 0.2
    save_dir: str, masks save path, a folder not a file path,
    e.g. './indian_pines'

    Returns
    -------
    train_mask: ndarray, H*W
    val_mask: ndarray, H*W
    test_mask: ndarray, H*W

    """

    train = get_each_class_num(target, train_num)
    val = get_each_class_num(target, val_num)

    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        train_i = train[i - 1]   # 每一类的训练个数
        val_i = val[i - 1]          # 每一类的验证个数

        np.random.shuffle(idx)
        train_idx = idx[:train_i]
        val_idx = idx[train_i:train_i + val_i]
        test_idx = idx[train_i + val_i:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        folder_name = 'masks_{}_{}'.format(train_num, val_num)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'),
                    {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'),
                    {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'),
                    {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


def load_masks(root_dir, target, train_prop, val_prop):

    train_prop = int(train_prop) if train_prop >= 1 else train_prop
    val_prop = int(val_prop) if val_prop >= 1 else val_prop

    # 添加一个文件夹在root_dir路径下
    masks_dir = os.path.join(root_dir, 'masks_{}_{}'.format(train_prop,
                                                            val_prop))
    try:
        # 这边是如果路径中有mask的话，直接用loadmat函数获得mask,然后return 三个mask
        train_mask = sio.loadmat(os.path.join(masks_dir, 'train_mask.mat'))['train_mask']
        val_mask = sio.loadmat(os.path.join(masks_dir, 'val_mask.mat'))['val_mask']
        test_mask = sio.loadmat(os.path.join(masks_dir, 'test_mask.mat'))['test_mask']
    except IOError:
        import platform
        if platform.python_version()[0] == '2':
            input_func = eval('raw_input')
        else:
            input_func = eval('input')

        # 路径中没有mask的话，输入y,调用get_proportional_masks函数生成mask
        flag = input_func('Prepare dataset, masks file not found! If produce a '
                          'new group of masks? [y/n] >> ')
        while True:
            if flag == 'y':
                if train_prop < 1:
                    train_mask, val_mask, test_mask = \
                        get_mask(target, train_prop, val_prop, save_dir=root_dir)
                else:
                    train_mask, val_mask, test_mask = \
                        get_mask(target, train_prop, val_prop, save_dir=root_dir)
                break
            elif flag == 'n':
                print('Program has terminated.')
                sys.exit()
            else:
                flag = input_func('Unknown character! please enter again >> ')

    return train_mask, val_mask, test_mask


def fixed_class_num_mask(target, each_class_num, val_total_num, min_num=5):

    class_num = target.max().astype(int)  # 类别的数量  16类

    target_count = target[target != 0]     # 所有有标签的样本个数  去除背景
    assert class_num * each_class_num < len(target_count)

    num_i_ = pd.value_counts(target_count).sort_index().values  # 每一类的总样本数,用于和固定每类样本的个数比较
    val = get_each_class_num(target, val_total_num)
    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        if num_i_[i-1] < each_class_num:
            class_i_num_train = min(num_i_[i-1], min_num)
            class_i_num_val = val[i - 1]

            np.random.shuffle(idx)
            train_idx = idx[:class_i_num_train]
            val_idx = idx[class_i_num_train:class_i_num_train + class_i_num_val]
            test_idx = idx[class_i_num_train + class_i_num_val:]

            train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
            val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
            test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

        else:
            class_i_num_val = val[i - 1]

            np.random.shuffle(idx)
            train_idx = idx[:each_class_num]
            val_idx = idx[each_class_num:each_class_num+class_i_num_val]
            test_idx = idx[each_class_num + class_i_num_val:]

            train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
            val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
            test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    return train_mask, val_mask, test_mask


def get_sample(data, target, mask, patch_size=13):
    # padding
    width = patch_size//2
    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'reflect')
    target = np.pad(target, ((width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')

    target = target * mask
    target = target[target != 0]-1

    patch_data_center = np.zeros((target.size, data.shape[0], patch_size, patch_size))
    patch_data_agumentation = np.zeros((target.size, data.shape[0], patch_size, patch_size))

    indx = np.argwhere(mask == 1)

    # label in center
    for i, col in enumerate(indx):
        patch_center = data[:, col[0]-width:col[0]+width+1, col[1]-width:col[1]+width+1]
        patch_data_center[i, :, :, :] = patch_center


    return patch_data_center, target


def get_fullmask_sample(data, target, mask):
    # data[target == 0] = 0
    # data = data.transpose(2, 0, 1)
    data = data * mask
    data = np.expand_dims(data, axis=0)
    # target = (target - 1) * mask
    target = target * mask
    target = np.expand_dims(target, axis=0)
    return data, target


def get_all_patches(data, patch_size):
    """get patches of all data points in the HSI data

    Parameters
    ----------
    data: ndarray, C*H*W
    patch_size: int, e.g. 13

    Returns
    -------
    patch_data: ndarray, N*C*P*P, N=H*W, P is patch size

    """
    width = patch_size // 2
    mask = np.ones((data.shape[1], data.shape[2]))  # 145*145

    patch_data = np.zeros((data.shape[1] * data.shape[2], data.shape[0],
                           patch_size, patch_size), dtype='float32')
    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')

    index = np.argwhere(mask)
    for i, loc in enumerate(index):
        patch_data[i, :, :, :] = data[:, loc[0] - width:loc[0] + width + 1,
                                 loc[1] - width:loc[1] + width + 1]

    return patch_data


def get_all_data(data):

    all_data = np.expand_dims(data, axis=0)

    return all_data


def batch_test(model, dataset, batch_size):
    model.eval()
    torch.cuda.empty_cache()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output = None
    for idx, batch_data in enumerate(dataloader):
        with torch.no_grad():
            # batch_output = model(batch_data[0].cuda()).cpu().data
            batch_output = model(batch_data[0].cuda())
        if idx == 0:
            output = batch_output
        else:
            output = torch.cat((output, batch_output), dim=0)

    pred = torch.max(output, dim=1)[1].data.cpu().numpy()
    target = dataset.target.cpu().numpy()
    pred = pred[target != 0]
    target = target[target != 0]
    oa = float((pred == target).sum()) / float(len(target))
    return oa


def test(model, dataset, batch_size):
    model.eval()
    torch.cuda.empty_cache()
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    for idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            labels = target.cuda()
            pre = model(data.cuda())
            output = torch.max(pre, dim=1)[1].data.cpu().numpy()
        # prediction = output + 1  # 1-16
            prediction = output[labels.cpu().numpy() != 0]
            labels = labels[labels.cpu().numpy() != 0]
        # total = labels.size(1) * labels.size(2)
            total = labels.shape[0]
            crroret = (prediction == labels.cpu().numpy()).sum().item()
            oa = crroret/total
    return oa


def predict(model, predict_dataset, batch_size):

    model.eval()
    model.cuda()

    dataloader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)
    # enumerate(dataloader)返回的是索引idx和数据batch_data
    for idx, batch_data in enumerate(dataloader):
        torch.cuda.empty_cache()
        with torch.no_grad():
            batch_output = model(batch_data.cuda()).cuda().data
            #batch_output, _ = model(batch_data.cuda())
        if idx == 0:
            output = batch_output
        else:
            output = torch.cat((output, batch_output), dim=0)  # 将两个tensor竖着拼在一起


    output = torch.max(output, dim=1)[1].cpu().numpy() + 1

    return output


def train(model, epoch, train_dataset, val_dataset,
          parameter, batch_size):

    criterion = parameter.get('criterion')
    optimizer = parameter.get('optim')
    lr = parameter.get('lr')
    lr_interval = int(epoch * 0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # train_dataloader = DataLoader(train_dataset, 32, shuffle=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    # 开始训练
    loss_log = []
    val_loss_log = []
    valacc_log = []
    best_acc = 0
    best_model = None
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        mse = 0

        val_tmp_loss = 0
        val_total_loss = 0
        k = 0
        total_valacc = 0
        for i, (data, target) in enumerate(train_dataloader):

            torch.cuda.empty_cache()  # data 小patch
            data, target = data.cuda(), target.cuda()

            output = model(data)  # shape of global_rep1 [B,D]  the same global_rep2
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 2 == 0:
                #train_acc = batch_test(model, train_dataset, batch_size=batch_size)
                val_acc = batch_test(model, val_dataset, batch_size=batch_size)
                k += 1
                total_valacc += (val_acc * 100)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = deepcopy(model)
            total_loss += loss.item()
        #
        print(time.ctime(), '[Epoch: %d]  [loss avg: %.5f]  [current loss: %.5f]  [current val acc: %.2f]' %
              (epoch + 1,total_loss / (i + 1), loss.item(), total_valacc / k))
        #
        loss_log.append(total_loss / (i + 1))
        valacc_log.append(total_valacc / k)
        if (epoch+1) % lr_interval == 0:
            scheduler.step()

    save_dir = './model_save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = "best-model" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
    model_dir = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_dir)

    print('Finished Training')

    return best_model, valacc_log, loss_log


def Train(model, epoch, train_dataset, val_dataset, parameter, batch_size):

    # model.train()
    lr = parameter.get('lr')
    criterion = parameter.get('criterion', nn.CrossEntropyLoss)
    optimizer = parameter.get('optim', torch.optim.Adam(model.parameters(), lr=lr))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00003)
    # train_dataloader = DataLoader(train_dataset, 32, shuffle=True)
    lr_interval = int(epoch/10)
    best_acc = 0
    best_model = None
    best_record = None
    state_dict = None
    best_state = None
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
    train_ACC = []
    val_ACC = []
    LOSS = []

    T_std = []
    V_std = []
    L_std = []
    for epoch in range(epoch):

        model.train()

        train_std = []
        val_std = []
        loss_std = []

        sum_loss = 0
        sum_train_acc = 0
        sum_val_acc = 0
        for i, (data, target) in enumerate(train_dataloader):

            torch.cuda.empty_cache()
            data, target = data.cuda(), target.cuda()
            prediction = model(data)  # [num, 16]
            # pdb.set_trace()
            #_, out = torch.max(prediction, dim=1)
            cost = criterion(prediction, target)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if i % 1 == 0:
                #train_acc = batch_test(model, train_dataset, batch_size=batch_size)
                val_acc = batch_test(model, val_dataset, batch_size=batch_size)

            #val_acc = batch_test(model, val_dataset, batch_size=batch_size)
            #test_acc = test(model, test_dataset, device)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = deepcopy(model)
                    state_dict = deepcopy(model.state_dict())
                    best_record = {'epoch': epoch, 'loss': cost.item(), 'model': model,
                                   'val_acc': val_acc}

                print('epoch: %-4s, loss: %.6f, val_acc: %.6f' %
                    (epoch + 1, cost.item(), val_acc))

        if (epoch+1) % lr_interval == 0:
            scheduler.step()
        # CosineLR.step()
        print("lr: %.8f" % optimizer.param_groups[0]['lr'])

    print('*'*20, 'best_model', '*'*20)
    print('epoch: %-4s, loss: %.6f,val acc: %.6f'
          % (best_record['epoch'] + 1, best_record['loss'], best_record['val_acc']))

    save_dir = './model_save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = "best-model" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
    model_dir = os.path.join(save_dir, model_name)
    #torch.save(model.state_dict(), model_dir)
    print('Finished Training')
    torch.save(state_dict, model_dir)
    #output = {'best_model': best_model, 'best_record': best_record}
    #prediction_ = prediction.argmax(1)
    return best_model


def pred_by_model(model, dataset, config):
    result = None
    input_shape = (1,) + dataset.data.shape[1:]
    print(input_shape)
    # model = get_net(input_shape, config["model"])
    # model.load_state_dict(torch.load(model_path))
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    for i, one_batch in enumerate(dataloader):
        print("\rPredicting Batch: {0}/{1}, {2:.2f}%".format(i, len(dataloader), (i / len(dataloader)) * 100),
              end="")
        data = one_batch.cuda()
        pred = model(data)
        pred = pred.detach().cpu().numpy()
        if i == 0:
            result = pred
        else:
            result = np.concatenate((result, pred), axis=0)
    print()
    return result + 1




def compute_accuracy(pre, target):
    with torch.no_grad():
        # target = torch.tensor(target)
        # pre = np.squeeze(pre)
        pre = pre[target != 0]
        target = target[target != 0]
        total = target.shape[0]
        crrort = (pre == target).sum().item()
        oa = float(crrort)/float(total)

    all_acc_Path = "./" + "acc_info/" + "acc_info" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"

    if not os.path.exists(all_acc_Path):
        os.makedirs(all_acc_Path)

    output = open(all_acc_Path + 'OA.csv', 'w', encoding='gbk')
    output.write('OA\n')
    output.write(str(oa))  # write函数不能写int类型的参数，所以使用str()转化
    output.close()

    return oa


class common_dataset(Dataset):
    """
    Args:
        data: ndarray, [1, 200, 13, 13]
        target: ndarray, [100,]
    """

    def __init__(self, data, target, cuda=True):

        super(common_dataset, self).__init__()  # 类common_dataset继承自Dataset,并做初始化

        self.data = torch.from_numpy(data).float()  # Tensor中的默认数据类型为浮点型(float)
        self.target = torch.from_numpy(target).long()

        if cuda is True:  # 采用GPU加速模型的训练
            self.data = torch.from_numpy(data).float().cuda()
            self.target = torch.from_numpy(target).long().cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

class common_valdataset(Dataset):
    """
    Args:
        data: ndarray, [1, 200, 13, 13]
        target: ndarray, [100,]
    """

    def __init__(self, data, target, cuda=True):

        super(common_valdataset, self).__init__()  # 类common_dataset继承自Dataset,并做初始化

        self.data = torch.from_numpy(data).float()  # Tensor中的默认数据类型为浮点型(float)
        self.target = torch.from_numpy(target).long()

        if cuda is True:  # 采用GPU加速模型的训练
            self.data = torch.from_numpy(data).float().cuda()
            self.target = torch.from_numpy(target).long().cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

class MyDataset(Dataset):
    def __init__(self, init_data, init_target):
        self.data = init_data
        self.target = init_target
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, indx):
        return self.data[indx], self.target[indx]


class PredictionData(Dataset):

    def __init__(self, all_data, cuda=True):

        super(PredictionData, self).__init__()

        # 类predict_dataset继承自Dataset,并做初始化

        self.data = torch.from_numpy(all_data).float()
        # Tensor中的默认数据类型为浮点型(float)

        if cuda is True:     # 采用GPU加速模型的训练
            #self.data = torch.cuda.FloatTensor(all_data)
            self.data = torch.from_numpy(all_data).float().cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item]



def check_path(path):
    """check a file path or folder, if the dirname is not find then creat it
    and return the input, for a file path, if it is exist then return a path
    string that obtained by adding a number suffix to the input, else return
    the input

    Parameters
    ----------
    path: str, a file path or a folder

    Returns
    -------
    path: str, a file path or a folder

    """

    path_tuple = os.path.splitext(path)

    if path_tuple[1] == '':
        path_dirname = path_tuple[0]
    elif path_tuple[1].split('.')[1].isdigit():
        path_dirname = path
    else:
        path_dirname = os.path.dirname(path)

    if not os.path.exists(path_dirname):
        print("Check path: '%s', folder '%s' not found! will create it ..." %
              (path, path_dirname))
        os.makedirs(path_dirname)

    if os.path.isfile(path):
        if os.path.exists(path):
            i = 1
            while True:
                path = '%s(%s)%s' % (path_tuple[0], i, path_tuple[1])
                if not os.path.exists(path):
                    break
                i += 1

    return path



def display_map_and_save(map, save_dir, show=False):

    # map: ndarray: h*w
    # save_dir: str, e.g. './classification_map.png'

    cmap = colors.ListedColormap(indian_colors, 'indexed')
    norm = colors.Normalize(vmin=0, vmax=16)
    height, width = map.shape
    fig, ax = plt.subplots()
    ax.imshow(map, cmap=cmap, norm=norm)
    plt.axis('off')
    fig.set_size_inches(width / 100.0, height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0, 0)

    plt.savefig(save_dir, dpi=800)
    if show:
        plt.show()
    plt.close()

indian_colors = ['#FFFFFF', '#B0C4DE', '#E9967A', '#AFEEEE', '#BC8F8F', '#66CDAA',
                '#7B68EE', '#FF7F50', '#5F9EA0', '#3CB371', '#DA70D6', '#90EE90',
                '#4682B4', '#FAA460', '#9ACD32', '#6B8E23', '#6e7955']

pavia_colors = ['#000000', '#533900', '#b0c4de', '#e9967a', '#EEE8AA',
                '#bc8f8f', '#66cdaa', '#9ACD32', '#7b68ee', '#030e73']


salinas_colors = ['#FFFFFF', '#895b8a', '#006e54', '#e6b422', '#8b968d', '#5a544b',
              '#ad7e4e', '#ee827c', '#008899', '#cca4e3', '#6b6882', '#fef263',
              '#028760', '#727171', '#a16d5d', '#00a497', '#ad7d4c']

Houston_color = ['#000000', '#934b43', '#f1d77e', '#9394e7', '#b1ce46', '#5f97d2', '#ee7a6d'
                 , '#62e197', '#9dc3e7', '#ffbe7a', '#b883d3', '#458b00', '#7f7f7f', '#8b8378',
                 '#4b3621', '#3eb489', '#808000', '#69359c', '#2c1608', '#d2b48c', '#f49ac2']

WHU_Longkou = ['#000000', '#726d40', '#88cb7f', '#316745', '#312520', '#a2d7dd', '#546E7A', '#007bbb', '#eaedf7', '#b3ada0']

WHU_Honghu = ['#000000', '#934b43', '#f1d77e', '#9394e7', '#b1ce46', '#5f97d2', '#ee7a6d'
                 , '#62e197', '#9dc3e7', '#ffbe7a', '#b883d3', '#458b00', '#7f7f7f', '#8b8378',
                 '#4b3621', '#3eb489', '#808000', '#69359c', '#2c1608', '#d2b48c', '#f49ac2', '#726d40', '#88cb7f']

WHU_Hanchuan = ['#000000', '#f47983', '#928178', '#7b8d42', '#595857', '#028760', '#fbca4d', '#f7b977',
                '#a0d8ef', '#c4a3bf', '#006e54', '#00a497', '#cd5e3c', '#a2d7dd', '#e0c38c', '#93b69c', '#1e50a2']

ice_classification = ['#FFFFFF', '#00ffc5', '#beffe8', '#97dbf2', '#089cd5']

def get_confusion_matrix(pred, target):

    conf_m = confusion_matrix(target.ravel(), pred.ravel(),
                              labels=range(1, int(target.max()) + 1))

    all_acc_Path = "./" + "acc_info/" + "acc_info" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"

    if not os.path.exists(all_acc_Path):
        os.makedirs(all_acc_Path)

    output = open(all_acc_Path + 'confusion.csv', 'w', encoding='gbk')
    output.write('confusion\n')
    for i in range(len(conf_m)):
        for j in range(len(conf_m[i])):
            output.write(str(conf_m[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')
        output.write('\n')  # 写完一行立马换行
    output.close()

    plt.matshow(conf_m, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(conf_m)):
        for j in range(len(conf_m)):
            plt.annotate(conf_m[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.tick_params(labelsize=15)  # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.savefig(all_acc_Path + "confusion.png", dpi=600)

    return conf_m


def compute_kappa_coef(pred, target):

    conf_m = confusion_matrix(target.ravel(), pred.ravel(),
                              labels=range(1, int(target.max()) + 1))
    #conf_m = get_confusion_matrix(pred, target)
    total = np.sum(conf_m)
    pa = np.trace(conf_m) / float(total)
    pe = np.sum(np.sum(conf_m, axis=0) * np.sum(conf_m, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    #
    # all_acc_Path = "./" + "acc_info/" + "acc_info" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"
    #
    # if not os.path.exists(all_acc_Path):
    #     os.makedirs(all_acc_Path)
    #
    # output = open(all_acc_Path + 'kappa.csv', 'w', encoding='gbk')
    # output.write('kappa\n')
    # output.write(str(kappa))  # write函数不能写int类型的参数，所以使用str()转化
    # output.close()

    return kappa

# def compute_accuracy_each_class(pred, target):
#
#     # pred: ndarray, H*W
#     # target: ndarray, H*W
#     # mask: ndarray, H*W
#
#     pred = pred.copy()
#     pred[target == 0] = 0
#     # target = target.copy()
#     # pred = pred*mask
#     # target = target*mask
#
#     accuracy = np.zeros(target.max().astype(np.int), dtype='float64')
#     for i in range(target.max().astype(np.int)):
#         if len(target[target == i + 1]) == 0:
#             accuracy[i] = 0
#         else:
#             accuracy[i] = float((pred[target == i + 1] == target[target == i + 1]).sum()) / \
#                      float(len(target[target == i + 1]))
#
#     return accuracy

def compute_accuracy_each_class(pred, target, mask):

    # pred: ndarray, H*W
    # target: ndarray, H*W
    # mask: ndarray, H*W

    pred = pred.copy()
    pred[target == 0] = 0
    target = target.copy()
    pred = pred*mask
    target = target*mask

    accuracy = np.zeros(target.max().astype(np.int), dtype='float64')
    for i in range(target.max().astype(np.int)):
        if len(target[target == i + 1]) == 0:
            accuracy[i] = 0
        else:
            accuracy[i] = float((pred[target == i + 1] == target[target == i + 1]).sum()) / \
                     float(len(target[target == i + 1]))

    return accuracy



def all_acc_info(pred, target):

    #get_confusion_matrix(pred, target)

    compute_kappa_coef(pred, target)
    
    acc_each_class = compute_accuracy_each_class(pred, target)

    compute_accuracy(pred, target)

    #F1_score = f1_score(target.flatten(), pred.flatten(), average='weighted')

    all_acc_Path = "./" + "acc_info/" + "acc_info" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"

    if not os.path.exists(all_acc_Path):
        os.makedirs(all_acc_Path)

    class_acc = pd.DataFrame(list(acc_each_class))
    class_acc.to_csv(all_acc_Path + "class_acc.csv", mode="a", header=False, index=True)

    # score = open(all_acc_Path + 'f1_score.csv', 'w', encoding='gbk')
    # score.write('F1_ecore\n')
    # score.write(str(F1_score))  # write函数不能写int类型的参数，所以使用str()转化
    # score.close()

def get_one_object(arr, seed_coor):

    object_coor = [seed_coor]

    def neighbor_grow(seed_coor):
        i, j = seed_coor
        coor_neighbors = [[i - 1, j - 1], [i - 1, j], [i - 1, j + 1],
                          [i, j - 1], [i, j + 1],
                          [i + 1, j - 1], [i + 1, j], [i + 1, j + 1]]
        valid_neighbors = []
        for idx, [p, q] in enumerate(coor_neighbors):
            if arr[p, q] == arr[i, j] and [p, q] not in object_coor:
                valid_neighbors.append([p, q])
                object_coor.append([p, q])

        for i, j in valid_neighbors:
            neighbor_grow([i, j])

    neighbor_grow(seed_coor)
    object_coor = np.array(object_coor)

    return object_coor


def get_all_objects(arr, objects_coor):

    h, w = arr.shape
    i, j = 0, 0
    seed_coor = []
    while arr[i, j] == 0:
        j = j + 1 if j < w - 1 else 0
        i = i + 1 if j == 0 else i
        seed_coor = [i, j]
        if i == h:
            return objects_coor

    object_coor = get_one_object(arr, seed_coor)
    arr[object_coor[:, 0], object_coor[:, 1]] = 0
    objects_coor.append(object_coor)
    objects_coor = get_all_objects(arr, objects_coor)

    return objects_coor


def cut_one_object(object_coor):

    train_coor = []
    test_coor = []

    # consider objects that only have one pixel
    if len(object_coor) == 1:
        test_coor += object_coor
        return train_coor, test_coor

    # cut the object horizontally or vertically
    flag = randint(0, 1)
    if object_coor[:, 0].min() == object_coor[:, 0].max():
        flag = 1
    if object_coor[:, 1].min() == object_coor[:, 1].max():
        flag = 0

    if flag == 0:
        point = object_coor[:, 0].max()
        point_ = object_coor[:, 0].min()
        while len(test_coor) <= len(train_coor) and point != point_:
            train_coor = [coor for coor in object_coor if coor[0] < point]
            test_coor = [coor for coor in object_coor if coor[0] >= point]
            point -= 1
    else:
        point = object_coor[:, 1].max()
        point_ = object_coor[:, 1].min()
        while len(test_coor) <= len(train_coor) and point != point_:
            train_coor = [coor for coor in object_coor if coor[1] < point]
            test_coor = [coor for coor in object_coor if coor[1] >= point]
            point -= 1

    train_coor = np.array(train_coor)
    test_coor = np.array(test_coor)

    return train_coor, test_coor


def disjoint_sampling(gt):

    gt_pad = np.pad(gt, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    h, w = gt_pad.shape

    objects_coor = []
    objects_coor = get_all_objects(gt_pad.copy(), objects_coor)

    train_gt = np.zeros((h, w)).astype(int)
    test_gt = np.zeros((h, w)).astype(int)

    for object_coor in objects_coor:
        train_coor, test_coor = cut_one_object(object_coor)
        train_gt[train_coor[:, 0], train_coor[:, 1]] = \
            gt_pad[train_coor[0][0], train_coor[0][1]]
        test_gt[test_coor[:, 0], test_coor[:, 1]] = \
            gt_pad[test_coor[0][0], test_coor[0][1]]

    train_gt = train_gt[1:h - 1, 1:w - 1]
    test_gt = test_gt[1:h - 1, 1:w - 1]

    return train_gt, test_gt


# get train mask, val mask and test mask use 'train_gt' and 'test_gt', the
# validation samples will be extracted from test samples set

def get_train_mask(train_gt, train_num):

    train_mask = np.zeros(train_gt.shape).astype(int)
    class_num = train_gt.max()

    train = get_each_class_num(train_gt, train_num)

    for i in range(1, class_num + 1):
        idx = np.argwhere(train_gt == i)
        train_i = train[i - 1]

        np.random.shuffle(idx)
        train_idx = idx[:train_i]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1

    return train_mask


def get_val_test_mask(test_gt, val_num):

    # get some pixels as validation samples from 'test_gt', and the rest as test
    # samples

    val_mask = np.zeros(test_gt.shape).astype(int)
    test_mask = np.zeros(test_gt.shape).astype(int)
    class_num = test_gt.max()

    val = get_each_class_num(test_gt, val_num)

    for i in range(1, class_num + 1):
        idx = np.argwhere(test_gt == i)
        val_i = val[i - 1]

        np.random.shuffle(idx)
        val_idx = idx[:val_i]
        test_idx = idx[val_i:]

        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    return val_mask, test_mask

def compute_accuracy_from_mask(pred, target, mask):
    """compute accuracy using 2D predicted data and a 2D mask

    Parameters
    ----------
    pred: ndarray, H*W
    target: ndarray, H*W
    mask: one of train, validation and test masks, ndarray, H*W

    Returns
    -------
    accuracy: float

    """

    pred = pred.copy()
    pred[target == 0] = 0
    target = target.copy()
    pred = pred * mask
    target = target * mask

    pred = pred[target != 0]
    target = target[target != 0]
    accuracy = float((pred == target).sum()) / float(len(pred))

    return accuracy

class ResultOutput:
    """a class can output training results:
    1) output one result record(accuracies, parameters, ...) after each training
    2) plot accuracy curves and save
    3) display result maps and save

    Parameters
    ----------
    pred: predicted map, ndarray: H*W
    target: ground truth, ndarray: H*W
    params: parameters dictionary, dict, params must contains these items:
    'accuracy_save': ... (accuracy save path, like './results/accuracy.csv')
    'map_csv': ... (predicted map/accuracy curves graph/parameters file(.json)/... save path, is a folder,
    like './results/result_map/')
    'mask_dir': ... (sample masks(train/val/test) save path)
    """

    def __init__(self, pred, target, map, train_mask, val_mask, test_mask,params):

        self.pred = pred
        self.target = target
        self.map = map
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.params = params

        self.acc_save = params['accuracy_save']
        if os.path.exists(self.acc_save):
            self.acc_df = pd.read_csv(self.acc_save)
        elif not os.path.exists(os.path.dirname(self.acc_save)):
            os.makedirs(os.path.dirname(self.acc_save))
            self.acc_df = None
        else:
            self.acc_df = None

        self.map_save = params['map_save']
        if not os.path.exists(self.map_save):
            os.makedirs(self.map_save)

    def compute_accuracy_and_save(self, params_record):

        # params_record: parameters need to be recorded, list, e.g. ['train_prop', 'lr', 'epoch', 'var', ...]

        # mask_dir = self.params['mask_dir']
        # train_mask = sio.loadmat(os.path.join(mask_dir, 'train_mask.mat'))['train_mask']
        # val_mask = sio.loadmat(os.path.join(mask_dir, 'val_mask.mat'))['val_mask']
        # test_mask = sio.loadmat(os.path.join(mask_dir, 'test_mask.mat'))['test_mask']

        train_acc = compute_accuracy_from_mask(self.pred, self.target, self.train_mask)
        val_acc = compute_accuracy_from_mask(self.pred, self.target, self.val_mask)
        test_acc = compute_accuracy_from_mask(self.pred, self.target, self.test_mask)
        # pdb.set_trace()

        params_value = []
        for str in params_record:
            if str in self.params.keys():
                params_value.append(self.params[str])
            else:
                 print("save parameters error, parameter's name {} not found!".format(str))
                 sys.exit()

        kappa = compute_kappa_coef(self.pred, self.target)
        class_num = int(self.target.max())
        class_acc_name = ['test_class_{}'.format(i) for i in range(1, class_num + 1)]
        class_acc = compute_accuracy_each_class(self.pred, self.target, self.test_mask).tolist()

        field_names = params_record + ['train_accuracy', 'val_accuracy', 'test_accuracy', 'kappa'] \
                      + class_acc_name
        field_values = [params_value + [train_acc, val_acc, test_acc, kappa] + class_acc]
        df = pd.DataFrame(data=field_values, columns=field_names)

        if self.acc_df is not None:
            self.acc_df = self.acc_df.append(df, ignore_index=True)
            self.acc_df.to_csv(self.acc_save, index=False)
        else:
            df.to_csv(self.acc_save, index=False)
            self.acc_df = df

    def display_map_and_save(self):

        pred_save = check_path(os.path.join(self.map_save, 'predict.png'))
        label_save = check_path(os.path.join(self.map_save, 'label.png'))
        map_save = check_path(os.path.join(self.map_save, 'map.png'))
        display_map_and_save(self.pred, pred_save, show=False)
        display_map_and_save(self.target, label_save, show=False)
        display_map_and_save(self.map, map_save, show=False)


def metrics(prediction, target):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).
    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    #target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]

    results = {}

    cm = confusion_matrix(target.ravel(), prediction.ravel(),
                              labels=range(1, int(target.max()) + 1))

    # cm = confusion_matrix(
    #     target,
    #     prediction,
    #     labels=range(n_classes))

    results["Confusion_matrix"] = cm

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    results["TPR"] = TPR
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    results["prediction"] = prediction
    results["label"] = target

    return results


def show_results(results, save_dir, agregated=False):
    text = ""


    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1_scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion_matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion_matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1_scores"]
        kappa = results["Kappa"]

    label_values = list(range(1, cm.shape[0] + 1))
    #label_values = label_values[1:]
    plt.matshow(cm, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.tick_params(labelsize=10)  # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 12}) # 设置字体大小。
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.savefig(save_dir + "confusion.png", dpi=600)

    text += "Confusion_matrix :\n"
    text += str(cm)
    text += "\n"
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1_scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    print(text)
    output = open(save_dir + 'info.txt', 'w', encoding='gbk')
    output.write(text)
    output.close()