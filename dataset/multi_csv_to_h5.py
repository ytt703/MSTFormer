import pandas as pd
import os

# 定义 CSV 文件所在目录和输出 HDF5 文件名
# folder_path = "E:\\Users\\201\\PycharmProjects\\gjemnessund"  # 替换为你的 CSV 文件夹路径
# output_h5 = "gjemnessund.h5"  # 输出 HDF5 文件名
# folder_path = "E:\\Users\\201\\PycharmProjects\\bergsoysund"  # 替换为你的 CSV 文件夹路径
folder_path = "E:\\Users\\201\\bergsoysund" # 替换为你的 CSV 文件夹路径
output_h5 = "bergsoysund.h5"  # 输出 HDF5 文件名




# 创建 HDF5 文件并写入所有 CSV
with pd.HDFStore(output_h5, mode='w') as hdf_store:
    for idx, file in enumerate(os.listdir(folder_path)):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)  # 读取 CSV 文件
            key=os.path.splitext(file)[0].replace('-', '_')
            hdf_store[key] = df  # 存储数据到 HDF5 文件
            print(f"已添加 {file} 到 HDF5 文件，键名为: {key}")

print(f"所有 CSV 文件已成功存储到 HDF5 文件: {output_h5}")
