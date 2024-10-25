import numpy as np
import pandas as pd
import sys
sys.path.append("/opt/data/private/ljx/plf/qos_mi/CLASS")
import os
import math
from CLASS.data_util import data_util
# from CLASS.data_process import data_process
from CLASS.data_process_base import data_process
# from CLASS.validation import validation
from CLASS.feature_process import feature_process

from CLASS.loss_function import CBloss_xgb
from CLASS.LOG import make_print_to_file
import xgboost as xgb
import warnings

# 创建对象
my_data_util = data_util()
my_data_process = data_process()
my_feature_process = feature_process()

import debugpy
def test():
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass
test()

# 读取数据
def get_data_raw(path):
    '''
    Args:
        path: 数据集路径
    Returns:
        data，合并好的数据集合
    '''
    dfs ,path_name = [],[]
    for subdir, dirs, files in os.walk(path):
        #     print(subdir, dirs, files)
        #     print(len(dirs),len(files))
        for file in files:
            # 构建完整的文件路径
            file_path = os.path.join(subdir, file)
            # 检查文件格式（例如：只处理CSV文件）
            if file_path.endswith('.csv'):
                # 读取CSV文件内容
                subpath_name=os.path.basename(file_path)
                df = pd.read_csv(file_path)
                # 数据预处理,删除重复值
                df = df.drop_duplicates(subset=['time'], keep='first').reset_index(drop=True)
                # 删除第一行缺失数据
                df=df.drop(index=0).reset_index(drop=True)
                df = my_feature_process.feature_data_method(df)
                # 预处理完成之后，将数据放在dfs里面
                dfs.append(df)
                path_name.append(subpath_name)
    return dfs,path_name

# 特征工程
def data_file_process_NN(Data_list):
    """
    :param Data_list:
    """
    Data_list_featured = []
    for data in Data_list:
        # 处理time列
        data['time'] = pd.to_datetime(data['time'])
        data = data.sort_values('time')

        # 缺失值处理，根据提供的每个特征的默认值，将默认值填充为有效值, 这里将pic删除了
        miss_values_racongize_list = [
                'avgRtt','avgRttVar','avgRcvRtt' ,'sentCount', 'recvCount',
                'lostCount','retans','retransmit', 'unacked','totalretrans'
        ]

        data = my_data_process.Missing_value_process_method2(data, miss_values_racongize_list)
        # 待补充。。。异常值处理，根据提供的每个特征的异常值范围，将异常值替换为有效值

        # 数据拆分，时间间隔超于60s的拆分，保留拆分后数据长度>=12   (输入10+输出2/3)
        split_dfs=my_data_process.datatime_split(data) 
        for df_split in split_dfs:
            Data_list_featured.append(df_split)

    return Data_list_featured

def opearte_data(data_list):
    """
    :param data_list:
    """
    data=my_data_util.concat_Data(data_list)
    data = data.drop('time', axis=1)
    return data

# 统计卡顿值
def count_datanum(df_all):
    sum=0
    stuck_num=0
    for df in df_all:
        sum+=df.shape[0]
        stuck_num+=df['dataStall'].sum()
    return sum,stuck_num

root_path='/opt/data/private/ljx/plf/qos_mi/Data/BJ_Subway_DataSet'

log_file = make_print_to_file(path='./log')

#1、读取数据 dfs,path_name
Data_list,path_name =get_data_raw(root_path)

#2、对每一个文件进行缺失值填充以及特征工程
Data_list_featured = data_file_process_NN(Data_list)

# 5.8  划分训练集，取每个文件的前60% 20% 20%
test_size=0.2
validation=False
if validation:
    #训练集、测试集、验证集的划分比例
    data_list_train, data_list_test, data_list_validation=my_data_util.split_train_test_NN(Data_list_featured)
else:
    data_list_train,data_list_test=my_data_util.split_train_test(Data_list_featured,test_size)

data_list_train=opearte_data(data_list_train)
data_list_test=opearte_data(data_list_test)
# data_list_validation=opearte_data(data_list_validation)

# 5、划分特征、标签
train_x, train_y = my_data_util.split_feature_label(data_list_train)
test_x, test_y = my_data_util.split_feature_label(data_list_test)
# validation_x, validation_y = my_data_util.split_feature_label(data_list_validation)

# 训练模型
# 6、由CB LOSS的方法确定加权交叉熵损失函数的权重。
# per_class_num_glory = [train_y.shape[0] - train_y.sum(), train_y.sum()]
# CB_loss_glory = CBloss_xgb(per_class_num_glory, beta=0.9999)

model = xgb.XGBClassifier(importance_type='gain'
                               # , learning_rate=0.11357142857142856
                               # , scale_pos_weight=CB_loss_glory.get_alpha()
                               )

model.fit(train_x, train_y)
neg_f1, pos_f1 = my_data_util.compute_f1(model, test_x, test_y)
acc_list, recall_list, precision_list  = my_data_util.compute_metrics(model, test_x, test_y)

print("validation:",validation,"test_size:",test_size)
print('训练集：', train_x.shape[0], train_y.sum())
print('测试集：', test_x.shape[0], test_y.sum())
print('卡顿f1：{}'.format(round(neg_f1[8], 4)), '不卡顿f1:{}'.format(round(pos_f1[8], 4)))
print('准确度acc：{}'.format(round(acc_list[8], 4)), '召回率recall：{}'.format(round(recall_list[8], 4)), '精确度precision：{}'.format(round(precision_list[8], 4)))
log_file.close()
