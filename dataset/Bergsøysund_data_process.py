import pandas as pd
import h5py
import numpy as np
import re
from datetime import datetime
# 读取 HDF5 文件
file_path = "data_2Hz.h5"
with h5py.File(file_path, 'r') as h5file:

    # print("Keys at the root level:", list(h5file.keys()))
    root=list(h5file.keys())
    for i in range(1,len(root)):
        # print(root[i])
        columns=[]
        data_matrix_list=[]
        r1=list(h5file[root[i]].keys()) #acceleration displacement wave wind
        # print(r1)
        if len(r1) == 4:
            for ii in r1:
                # print(r1)
                r2=list(h5file[root[i]][ii].keys()) #G5 east west wind
                for iii in r2:
                    if iii == 'A1' or iii == 'A2' or iii == 'A3' or iii == 'A4' or iii == 'A5' or iii == '4N':
                        r3=list(h5file[root[i]][ii][iii].keys()) #x y z
                        for iiii in r3:
                            # print(h5file[root[i]][ii][iii][iiii])
                            data_vector=h5file[root[i]][ii][iii][iiii][:]
                            data_matrix_list.append(data_vector)
                            columns.append(ii+iii+iiii)
            data_matrix=np.asarray( data_matrix_list)
            data_matrix=np.transpose(data_matrix, axes=( 1, 0))
            # df = pd.DataFrame(data_matrix, columns=columns)
            # print(np.isnan(data_matrix).sum())
            nan_count = np.isnan(data_matrix).sum()
            if nan_count == 0:
                # print("yes",data_matrix.shape)
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', root[i])
                # print(match)
                date_str = match.group(0)  # '2013-11-27_09-02-54'
                # 将其转换为 datetime 对象
                datetime_obj = datetime.strptime(date_str, '%Y-%m-%d_%H-%M-%S')
                # 转换为目标格式的字符串（可选）
                # formatted_date = datetime_obj.strftime('%Y/%m/%d %H:%M:%S')
                date_range = pd.date_range(start=datetime_obj, periods=data_matrix.shape[0], freq='500L')
                df = pd.DataFrame(data_matrix,index=date_range ,columns=columns)
                df.index.name = 'date'  # 为索引命名
                print(i)
                df.to_csv(root[i]+'.csv')

