import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import glob
warnings.filterwarnings('ignore')
'''
root_path='dataset'
data_path='ETTh1.csv'


'''
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_gjemnessund(Dataset):
    def __init__(self, root_path='./dataset', flag='train', size=None,
                 features='S', data_path='gjemnessund.h5',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.data_x = []
        self.data_y = []
        self.data_stamp = []
        self.stride = 16
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # csv_files = glob.glob(os.path.join(self.root_path, "**", "*.csv"), recursive=True)
        with pd.HDFStore(os.path.join(self.root_path,self.data_path)) as hdf_store:
            csv_files=hdf_store.keys()
            train_data_all = []  # 用于全局训练数据拟合
            segments = []  # 存储所有片段的原始数据和索引
            for i in range(len(csv_files)):
                    # df_raw = pd.read_csv(csv_file)
                df_raw = hdf_store[csv_files[i]]
                cols = list(df_raw.columns)
                if self.features == 'S':
                    cols.remove(self.target)
                cols.remove('date')
                # print(cols)
                num_train = int(len(df_raw) * (0.5 if not self.train_only else 1))
                num_test = int(len(df_raw) * 0.25)
                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.features == 'M' or self.features == 'MS':
                    df_raw = df_raw[['date'] + cols]
                    cols_data = df_raw.columns[1:]
                    df_data = df_raw[cols_data]
                elif self.features == 'S':
                    df_raw = df_raw[['date'] + cols + [self.target]]
                    df_data = df_raw[[self.target]]

                train_data_all.append(df_data)  # 收集训练数据
                segments.append((df_raw, df_data, border1s, border2s))  # 存储片段信息
            # 全局拟合标准化器
            train_data_all = pd.concat(train_data_all, axis=0)
            self.scaler.fit(train_data_all.values)

            for df_raw, df_data, border1s, border2s in segments:
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.scale:
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values

                # 时间戳处理
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                if self.timeenc == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], axis=1).values
                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)

                data_x = data[border1:border2]
                data_y = data[border1:border2]

                self.data_x.append(data_x)
                self.data_y.append(data_y)
                self.data_stamp.append(data_stamp)

    def __getitem__(self, idx):
        for segment_x,segment_y,segment_stamp in zip(self.data_x,self.data_y,self.data_stamp):
            segment_len = len(segment_x) - self.seq_len - self.pred_len + 1
            if segment_len>0:
                num_samples = (segment_len + self.stride - 1) // self.stride  # 当前片段样本数
                if idx < num_samples:
                    start_idx = idx*self.stride
                    seq_x = segment_x[start_idx : start_idx + self.seq_len]  # 输入序列特征
                    seq_y = segment_y[start_idx + self.seq_len-self.label_len : start_idx + self.seq_len + self.pred_len]  # 目标列
                    seq_x_mark = segment_stamp[start_idx:start_idx+self.seq_len]
                    seq_y_mark = segment_stamp[start_idx + self.seq_len-self.label_len:start_idx + self.seq_len+self.pred_len]
                    return seq_x, seq_y,seq_x_mark,seq_y_mark
                idx -= num_samples

    def __len__(self):
        length=0
        for segment_x in self.data_x:
            segment_len=len(segment_x) - self.seq_len - self.pred_len + 1
            if segment_len>0:
                length+=(segment_len+self.stride-1)//self.stride
        return length


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_bergsoysund(Dataset):
    def __init__(self, root_path='dataset', flag='train', size=None,
                 features='S', data_path='bergsoysund.h5',
                 target='OT', scale=True, timeenc=0, freq='s', train_only=False,stride=16):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.data_x = []
        self.data_y = []
        self.data_stamp = []
        self.stride=stride
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # csv_files = glob.glob(os.path.join(self.root_path, "**", "*.csv"), recursive=True)
        with pd.HDFStore(os.path.join(self.root_path,self.data_path)) as hdf_store:
            csv_files=hdf_store.keys()
            train_data_all = []  # 用于全局训练数据拟合
            segments = []  # 存储所有片段的原始数据和索引
            for i in range(len(csv_files)):
                    # df_raw = pd.read_csv(csv_file)
                df_raw = hdf_store[csv_files[i]]
                cols = list(df_raw.columns)
                if self.features == 'S':
                    cols.remove(self.target)
                cols.remove('date')
                # print(cols)
                num_train = int(len(df_raw) * (0.5 if not self.train_only else 1))
                num_test = int(len(df_raw) * 0.25)
                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.features == 'M' or self.features == 'MS':
                    df_raw = df_raw[['date'] + cols]
                    cols_data = df_raw.columns[1:]
                    df_data = df_raw[cols_data]
                elif self.features == 'S':
                    df_raw = df_raw[['date'] + cols + [self.target]]
                    df_data = df_raw[[self.target]]

                train_data_all.append(df_data)  # 收集训练数据
                segments.append((df_raw, df_data, border1s, border2s))  # 存储片段信息
            # 全局拟合标准化器
            train_data_all = pd.concat(train_data_all, axis=0)
            self.scaler.fit(train_data_all.values)

            for df_raw, df_data, border1s, border2s in segments:
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.scale:
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values

                # 时间戳处理
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                if self.timeenc == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], axis=1).values
                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)

                data_x = data[border1:border2]
                data_y = data[border1:border2]

                self.data_x.append(data_x)
                self.data_y.append(data_y)
                self.data_stamp.append(data_stamp)

        # with pd.HDFStore(os.path.join(self.root_path,self.data_path)) as hdf_store:
        #     csv_files=hdf_store.keys()
        # # with pd.HDFStore(file_path, mode='r') as hdf_store:
        # # matrices = []
        #     for i in range(len(csv_files)):
        #         # df_raw = pd.read_csv(csv_file)
        #         df_raw = hdf_store[csv_files[i]]
        #         cols = list(df_raw.columns)
        #         if self.features == 'S':
        #             cols.remove(self.target)
        #         cols.remove('date')
        #         # print(cols)
        #         num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        #         num_test = int(len(df_raw) * 0.2)
        #         num_vali = len(df_raw) - num_train - num_test
        #         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        #         border2s = [num_train, num_train + num_vali, len(df_raw)]
        #         border1 = border1s[self.set_type]
        #         border2 = border2s[self.set_type]
        #
        #         if self.features == 'M' or self.features == 'MS':
        #             df_raw = df_raw[['date'] + cols]
        #             cols_data = df_raw.columns[1:]
        #             df_data = df_raw[cols_data]
        #         elif self.features == 'S':
        #             df_raw = df_raw[['date'] + cols + [self.target]]
        #             df_data = df_raw[[self.target]]
        #
        #         if self.scale:
        #             train_data = df_data[border1s[0]:border2s[0]]
        #             self.scaler.fit(train_data.values)
        #             # print(self.scaler.mean_)
        #             # exit()
        #             data = self.scaler.transform(df_data.values)
        #         else:
        #             data = df_data.values
        #
        #         df_stamp = df_raw[['date']][border1:border2]
        #         df_stamp['date'] = pd.to_datetime(df_stamp.date)
        #         if self.timeenc == 0:
        #             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #             data_stamp = df_stamp.drop(['date'], 1).values
        #         elif self.timeenc == 1:
        #             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #             data_stamp = data_stamp.transpose(1, 0)
        #
        #         data_x = data[border1:border2]
        #         data_y = data[border1:border2]
        #         data_stamp = data_stamp
        #         # print(data_x.shape,data_y.shape,data_stamp.shape,type(data_x),type(data_y),type(data_stamp))
        #         self.data_x.append(data_x)
        #         self.data_y.append(data_y)
        #         self.data_stamp.append(data_stamp)
        #         # matrices.append(data_x)

    def __getitem__(self, idx):

        # print(idx)
        for segment_x,segment_y,segment_stamp in zip(self.data_x,self.data_y,self.data_stamp):
            segment_len = len(segment_x) - self.seq_len - self.pred_len + 1
            if segment_len>0:
                num_samples = (segment_len + self.stride - 1) // self.stride  # 当前片段样本数
                if idx < num_samples:
                    start_idx = idx*self.stride
                    seq_x = segment_x[start_idx : start_idx + self.seq_len]  # 输入序列特征
                    seq_y = segment_y[start_idx + self.seq_len-self.label_len : start_idx + self.seq_len + self.pred_len]  # 目标列
                    seq_x_mark = segment_stamp[start_idx:start_idx+self.seq_len]
                    seq_y_mark = segment_stamp[start_idx + self.seq_len-self.label_len:start_idx + self.seq_len+self.pred_len]
                    return seq_x, seq_y,seq_x_mark,seq_y_mark
                idx -= num_samples

    def __len__(self):
        length=0
        for segment_x in self.data_x:
            segment_len=len(segment_x) - self.seq_len - self.pred_len + 1
            if segment_len>0:
                length+=(segment_len+self.stride-1)//self.stride
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
