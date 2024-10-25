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
        # 异常值替换，这6个特征出现-1替换为0
        features_to_replace = [
                'deltaTcpFwmarkLossRate','deltaTcpKernelLossRate','deltaTcpFwmarkRttAvg',
                'deltaTcpKernelRttAvg','deltaDnsRttAvg','deltaDnsSuccessRate'
                ]
        data[features_to_replace]=data[features_to_replace].replace(-1,0)
        #异常值处理，根据提供的每个特征的异常值范围，将异常值替换为有效值
        my_data_process.abnormal_value_process_method1(data)

        # 数据拆分，时间间隔超于60s的拆分，保留拆分后数据长度>=12   (输入10+输出2/3)
        split_dfs=my_data_process.datatime_split(data) 
        for df_split in split_dfs:
            Data_list_featured.append(df_split)

    return Data_list_featured

# 创建特征窗口
def create_features(data,input_size, output_size, timestep):
    """
    创建特征窗口
    :param data: 输入数据
    :param input_size: 输入窗口大小
    :param output_size: 输出窗口大小
    :param timestep: 时间步长
    """
    data_feature,data_label = [], []
    for df in data:
        for i in range(0,len(df) - input_size- output_size,timestep):
            input_feature=df.iloc[i:i + input_size]
            input_feature = input_feature.drop('dataStall', axis=1)#去掉标签
            data_feature.append(input_feature)
            data_label.append(df.iloc[i + input_size:i + input_size + output_size]['dataStall'])

    data_feature = np.array(data_feature)   # 长为20的输入特征，无标签  #[256,20,50]
    data_label = np.array(data_label) 

    return data_feature, data_label

def drop_time(datalist):
    data_list=[]
    for df in datalist:
        data=df.drop('time', axis=1)
        data_list.append(data)
    return data_list

# 统计卡顿值
def count_datanum(df_all):
    sum=0
    stuck_num=0
    for df in df_all:
        sum+=df.shape[0]
        stuck_num+=df['dataStall'].sum()
    return sum,stuck_num

def opearate_data(root_path, input_size, output_size, timestep,test_size=0.2,validation=False):
    """
    数据预处理
    :param data: 输入数据
    """
    #1、读取数据 dfs,path_name
    Data_list,path_name =get_data_raw(root_path)
    print('数据读取结束------------------------------------》')
    #2、对每一个文件进行缺失值填充以及特征工程
    Data_list_featured = data_file_process_NN(Data_list)  
    print('特征工程结束----------》')

    Data_list_featured=drop_time(Data_list_featured)
    # 划分训练集，取每个文件的前60% 20% 20%
    print('数据集划分开始----------》')
    print('test_size:',test_size,'validation:',validation)
    if validation:
        #训练集、测试集、验证集的划分比例：是否需要验证集；6：2：2是深度学习常用的划分比例；xgboost不需要验证集
        data_list_train, data_list_test, data_list_validation=my_data_util.split_train_test_NN(Data_list_featured)
        train_x, train_y = create_features(data_list_train, input_size, output_size, timestep)
        test_x, test_y = create_features(data_list_test, input_size, output_size, timestep)
        validation_x, validation_y = create_features(data_list_validation, input_size, output_size, timestep)
    else:
        data_list_train,data_list_test=my_data_util.split_train_test(Data_list_featured,test_size)
        train_x, train_y = create_features(data_list_train, input_size, output_size, timestep)
        test_x, test_y = create_features(data_list_test, input_size, output_size, timestep)

    sum_train,stuck_num_train = count_datanum(data_list_train)
    # sum_valid, stuck_num_valid = count_datanum(data_list_validation)
    sum_test, stuck_num_test = count_datanum(data_list_test)
    print('训练集样本数：{},卡顿样本数为：{},卡顿比例：{}'.format(sum_train,stuck_num_train,f'{round((stuck_num_train/sum_train)*100,2)}%'))
    # print('验证集样本数：{},卡顿样本数为：{}'.format(sum_valid, stuck_num_valid))
    print('测试集样本数：{},卡顿样本数为：{},卡顿比例：{}'.format(sum_test, stuck_num_test, f'{round((stuck_num_test/sum_test)*100,2)}%'))

    print('训练、验证、测试划分结束------------------------------------》')
    return train_x, train_y, test_x, test_y



# xgboost中有两种模型，一种是回归模型，一种是分类模型。下面是两种模型的实现

# 定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# xgboost用回归模型
def model_reg(train_x, train_y, test_x, test_y):
    # 定义回归模型   回归任务使用 XGBRegressor 或 xgb.train，目标是预测连续值
    model = xgb.XGBRegressor(objective='reg:squarederror',
                                    n_estimators=1000   #提升轮数，即要构建的树的数量。更多的估计器可以提高性能，但也增加了过拟合的风险
                                # , learning_rate=0.11357142857142856
                                # , scale_pos_weight=CB_loss_glory.get_alpha()
                                )

    model.fit(train_x, train_y)
    # 预测
    y_pred = model.predict(test_x)#预测值输出的是真实值,将其转换为0-1之间
    print('预测值输出结束----------》',max(y_pred),min(y_pred))
    y_pred = sigmoid(y_pred)
    print(max(y_pred),min(y_pred))
    y_pred=(y_pred>0.5).astype(float)
    test_y=test_y.flatten()

    return y_pred,test_y

# xgboost中用分类模型
def model_cls(train_x, train_y, test_x, test_y):
    # 定义分类模型
    # 6、由CB LOSS的方法确定加权交叉熵损失函数的权重。
    per_class_num_glory = [train_y.shape[0] - train_y.sum(), train_y.sum()]
    CB_loss_glory = CBloss_xgb(per_class_num_glory, beta=0.9999)
    model = xgb.XGBClassifier(importance_type='gain'
                                # , learning_rate=0.11357142857142856
                                # , scale_pos_weight=CB_loss_glory.get_alpha()
                                )
    model.fit(train_x, train_y)
    # 预测
    y_pred = model.predict_proba(test_x)[:,1]
    y_pred = (y_pred>0.5).astype(float)
    test_y=test_y.flatten()

    return y_pred,test_y


root_path='/opt/data/private/ljx/plf/qos_mi/Data/BJ_Subway_DataSet'
log_file = make_print_to_file(path='./log')

input_size = 10
output_size = 1
timestep = 1
# 划分训练集、测试集
test_size = 0.4        #测试集占总数据集的比例
validation = False      #这里是否需要划分验证集

train_x, train_y, test_x, test_y = opearate_data(root_path, input_size, output_size, timestep,test_size,validation)

# 确保数据是二维的
train_x = train_x.reshape(train_x.shape[0], -1) #seq_len, input_size*dim
test_x = test_x.reshape(test_x.shape[0], -1) #seq_len, input_size*dim
print('数据集形状调整结束----------》')

# 选择模型model_cls；model_reg
y_pred,test_y=model_cls(train_x, train_y, test_x, test_y)
# 计算卡顿f1、不卡顿f1、准确度acc、召回率recall、精确度precision
neg_f1, pos_f1,acc_list, recall_list, precision_list  = my_data_util.compute_metrics_NN(y_pred, test_y)

print('卡顿f1：{}'.format(round(neg_f1[0], 4)), '不卡顿f1:{}'.format(round(pos_f1[0], 4)))
print('准确度acc：{}'.format(round(acc_list[0], 4)), '召回率recall：{}'.format(round(recall_list[0], 4)), '精确度precision：{}'.format(round(precision_list[0], 4)))

log_file.close()
