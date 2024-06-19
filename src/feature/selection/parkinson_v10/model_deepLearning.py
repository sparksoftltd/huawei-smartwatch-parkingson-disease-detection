# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import time
import logging
import os, sys
import psutil
import lightgbm as lgb
from datetime import datetime

from itertools import cycle
from sklearn import svm
from sklearn.metrics import *
# from sklearn.cross_validation import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

import My_CrossValidation
from numpy import *


# For plotting learning curve
# from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class My_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.LongTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, input_dim=174, output_dim=5):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class My_Model(nn.Module):
    def __init__(self, input_dim=174, output_dim=3, hidden_layers=1, hidden_dim=256):
        super(My_Model, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], \
           raw_x_test[:, feat_idx], y_train, y_valid


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.CrossEntropyLoss(reduction='mean')  # Define your loss function, do not modify this.
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        # writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # MADE：my GPU is outdated
device = 'cpu'

# configuration dictionary
config = {
    'seed': 1000,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 1000,  # Number of epochs.
    'batch_size': 256,  # 256
    'learning_rate': 1e-2,
    'early_stop': 100,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt',  # Your model will be saved here.

    # input_dim=174, output_dim=3, hidden_layers=1, hidden_dim=256)
    'input_dim': 174,  # 输入特征维度
    'hidden_dim': 256,  # 隐藏层特征维度
    'hidden_layers': 1,  # 隐藏层的层数
    'output_dim': 5  # num of class

}

result_dic_column = ['Activity', 'Mean precision', 'Std precision']
result_dic_record = []
repetition_num = 10

total_activity = 14
activity_idx_list = range(0, total_activity + 1)
# activity_idx_list = range(9, 9 + 1)


dataset_idx = 2  # 用第几个数据集
dataset_name = 'Dataset' + str(dataset_idx)  # 具体的文件夹

for activity_idx in activity_idx_list:
    activity_name = 'activity' + str(activity_idx)  # 在运行第几个活动的数据集
    print(activity_name + ' is running')

    file_name = dataset_name + '/' + activity_name + '.csv'  # 具体的文件

    folder_num = 10
    test_ratio = 0.1
    is_onehot = 0  # 是否需要onehot编码

    result_list = []


    # 生成结果的文件夹

    first_layer_dictionary = dataset_name + '_result'
    second_layer_dictionary = 'Repetition_' + str(repetition_num)
    third_layer_dictionary = 'DL' + str(config['hidden_layers'] + 1)
    final_path = first_layer_dictionary + '/' + second_layer_dictionary + '/' + third_layer_dictionary + '/'
    if not os.path.exists(final_path):
        os.makedirs(final_path)


    for i in range(repetition_num):
        print('\t' + activity_name + ' is running on ' + str(i + 1) + 'th repetition')
        X_train_list, y_train_list, \
        X_val_list, y_val_list, \
        X_train_total, y_train_total, \
        X_test, y_test, dataset_drop = My_CrossValidation.my_cross_validation(file_name, folder_num,
                                                                              test_ratio,
                                                                              is_onehot)  # prepare the dataset

        order_validation = 1  # 第几组validation
        X_train = X_train_list[order_validation - 1]
        y_train = y_train_list[order_validation - 1]
        X_val = X_val_list[order_validation - 1]
        y_val = y_val_list[order_validation - 1]

        # Print out the number of features.
        print(f'number of features: {X_train.shape[1]}')
        train_dataset, valid_dataset, test_dataset = My_Dataset(X_train, y_train), \
                                                     My_Dataset(X_val, y_val), \
                                                     My_Dataset(X_test)

        # Pytorch data loader loads pytorch dataset into batches.
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        # put your model and data on the same computation device.
        # input_dim=174, output_dim=3, hidden_layers=1, hidden_dim=256)
        model = My_Model(input_dim=config['input_dim'], output_dim=config['output_dim'], \
                         hidden_dim=config['hidden_dim'], hidden_layers=config['hidden_layers']).to(device)
        trainer(train_loader, valid_loader, model, config, device)

        model = My_Model(input_dim=config['input_dim'], output_dim=config['output_dim'], \
                         hidden_dim=config['hidden_dim'], hidden_layers=config['hidden_layers']).to(device)
        model.load_state_dict(torch.load(config['save_path']))
        test_acc = 0.0
        test_lengths = 0
        pred = np.array([], dtype=np.int32)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                features = batch
                features = features.to(device)

                outputs = model(features)
                _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
                pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

        final_precision = precision_score(y_test, pred, average='micro')
        print('\t', final_precision)
        result_list.append(final_precision)

    print('\n')
    # 记录每一个activity运行repetation_num次的平均值和方差
    result_record = [activity_idx, mean(result_list), std(result_list)]
    result_dic_record.append(result_record)
    # print(result_record)




    save_path_scatter = final_path + activity_name + '_result.csv'
    pd_result_list = pd.DataFrame(data=result_list).round(3)
    pd_result_list.to_csv(save_path_scatter, index=False)

final_result = pd.DataFrame(columns=result_dic_column, data=result_dic_record)
final_result = final_result.round(3)  # 保留三位有效数字


save_path = final_path + 'DL' + str(
    config['hidden_layers'] + 1) + '.csv'  # 存储路径
final_result.to_csv(save_path, index=False)  # 保存文件
