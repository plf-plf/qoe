import numpy as np
import pandas as pd
import sys
sys.path.append("/opt/data/private/ljx/plf/qos_mi/CLASS")
import os
import math
from CLASS.data_util import data_util
# from CLASS.data_process import data_process
from CLASS.data_process_online import data_process
# from CLASS.validation import validation
from CLASS.feature_process import feature_process
from CLASS.loss_function import CBloss_xgb
from CLASS.LOG import make_print_to_file
import xgboost as xgb
import json

# 创建对象
my_data_util = data_util()
my_data_process = data_process()

# 读取数据
def read_list(folder_path):
    all_data_list = []
    # 遍历文件夹中的所有文件
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                # 读取 Excel 文件
                df = pd.read_excel(file_path)
                # 将读取的数据添加到列表中
                all_data_list.append(df)
    elif folder_path.endswith('.xlsx'):
        # 读取单个 Excel 文件
        df = pd.read_excel(folder_path)
        all_data_list.append(df) 
    return all_data_list

# 读取inputStats数据，对数据进行初步规范
def collect_data(nested_list, collected_data):
    for element in nested_list:
        if isinstance(element, list):
            if all(not isinstance(i, list) for i in element):
                collected_data.append(element)
            else:
                collect_data(element,collected_data)
                
def get_data_raw( folder_path):
    data_list=[]
    # 重新定义特征
    new_columns=["hwLevel","mLastPredictionPci","foreground_info","time","mLastPredictionCid",
                "rat", "rsrp", "rsrq", "snr", "pci",
                "deltaTcpKernelLossRate" ,"deltaDnsRttAvg","deltaTcpKernelRttAvg" ,"dlRateBps" ,"ulRateBps",
                "retans", "totalretrans" ,"unacked" ,"retransmit" ,"lostCount", "avgRttVar" ,"minRtt","avgRtt",
                "DataStall","isVideoFront"]
    # 读取Excel文件
    all_data_list = read_list(folder_path)
    for df in all_data_list:        
        for json_df in df['b']:
            collected_data=[]
            df_initial=json.loads(json_df)
            df_0 = pd.DataFrame([df_initial])
            # 其他特征
            df_0=df_0.drop(columns=['inputStats'])
            df_0 = pd.DataFrame(np.repeat(df_0.values, 31, axis=0), columns=df_0.columns)
            # inputStats的特征
            df_1=df_initial["inputStats"]
            df_1=json.loads(df_1)
            collect_data(df_1, collected_data) 
            df_1=pd.DataFrame(collected_data)
            # 合并重新定义的特征
            df_merged = pd.concat([df_0, df_1], axis=1)
            df_merged.columns = new_columns     # 重新定义特征
            # 将标签放到最后
            columns_pre = list(df_merged.columns)
            columns_pre.append(columns_pre.pop(columns_pre.index('DataStall')))
            df_merged = df_merged.reindex(columns=columns_pre)
            # 将指定的列转换为整数格式
            df_merged['mLastPredictionPci'] = df_merged['mLastPredictionPci'].astype(int)
            df_merged['mLastPredictionCid'] = df_merged['mLastPredictionCid'].astype(int)
            data_list.append(df_merged)
    print("数据集总长度：",len(data_list))
    return data_list

# 对每一个文件进行缺失值填充以及特征工程
def data_file_process_NN( Data_list,foreground_info='all'):
    """
    :param Data_list:
    """
    Data_list_featured = []

    for data in Data_list:
        # 将foreground_info的链接替换为0和1
        data=my_data_process.foreground_info_regconize(data)
        # 将卡顿标签转换为0和1：线上采集数据是0表示卡顿，1表示正常，便于后续计算和与线下保持统一，将0和1互换
        # 此时就是1表示卡顿，0表示正常
        data['DataStall'] = 1 - data['DataStall']
        # 添加新列，用于标记是否为视频卡顿
        # data['dataStall'] = data.apply(lambda row: 1 if row['isVedioDataStall'] == 1 and row['isVideoFront'] == 1 else 0, axis=1)
        # print("渲染卡顿isVedioDataStall：",sum(df['isVedioDataStall']),"渲染状态且渲染卡顿",sum(df['dataStall'] ))
        # 区分场景前后台
        # foreground_info=0是抖音，foreground_info=1是虎牙,all是所有场景的放在一起不区分;
        if foreground_info != 'all':
            data = data[data['foreground_info'] == int(foreground_info)]

        # 删除无用特征
        miss_values_racongize_list = ['foreground_info','hwLevel']
        data = my_data_process.drop_feature(data, miss_values_racongize_list)
        
        # 异常值处理，根据提供的每个特征的异常值范围，将异常值替换为有效值
        # data = my_data_process.abnormal_value_process_method1(data)

        Data_list_featured.append(data)

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
            input_feature = input_feature.drop('DataStall', axis=1)#去掉标签
            data_feature.append(input_feature)
            data_label.append(df.iloc[i + input_size:i + input_size + output_size]['DataStall'])

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
        stuck_num+=df['DataStall'].sum()
    return sum,stuck_num

def opearate_data( root_path, input_size, output_size, timestep,foreground_info):
    """
    数据预处理
    :param data: 输入数据
    """
    #1、读取数据 dfs,path_name
    Data_list = get_data_raw(root_path)
    print('数据读取结束------------------------------------》')
    #2、对每一个文件进行缺失值填充以及特征工程
    Data_list_feature = data_file_process_NN(Data_list, foreground_info)  
    print('特征工程结束----------》')

    # 3、删除时间列
    Data_list_feature = drop_time(Data_list_feature)
    print('删除时间列结束----------》')

    # 划分训练集，取每个文件的前60% 20% 20%
    print('数据集划分开始----------》')
    print('test_size:',test_size,'validation:',validation)

    data_list_train = []
    data_list_validation = []
    data_list_test = []
    #  划分训练集，取所有文件数据（438个文件）的前60% 20% 20%
    num_train = math.ceil(len(Data_list_feature) * 0.6)
    num_validation = math.ceil(len(Data_list_feature) * 0.2)
    num_test = math.ceil(len(Data_list_feature) * 0.2)

    data_list_train = Data_list_feature[0:num_train]
    data_list_validation = Data_list_feature[num_train:num_train + num_validation]
    data_list_test = Data_list_feature[num_train + num_validation:num_train + num_test + num_validation]

    train_x, train_y = create_features(data_list_train, input_size, output_size, timestep)
    test_x, test_y = create_features(data_list_test, input_size, output_size, timestep)

    print('训练、验证、测试划分结束------------------------------------》')
    sum_train,stuck_num_train = count_datanum(data_list_train)
    sum_test, stuck_num_test = count_datanum(data_list_test)
    sum_valid, stuck_num_valid = count_datanum(data_list_validation)
    
    print('训练集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_train,stuck_num_train,f'{round((stuck_num_train/sum_train)*100,2)}%'))
    print('验证集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_valid, stuck_num_valid,f'{round((stuck_num_valid/sum_valid)*100,2)}%'))
    print('测试集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_test, stuck_num_test, f'{round((stuck_num_test/sum_test)*100,2)}%'))
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


root_path='/opt/data/private/ljx/plf/qos_mi/Data/10.12/result_1021.xlsx'
log_file = make_print_to_file(path='./log')

input_size = 10
output_size = 1
timestep = 1
# 划分训练集、测试集
test_size = 0.2         #测试集占总数据集的比例
validation = False      #这里是否需要划分验证集
foreground_info="all"

train_x, train_y, test_x, test_y = opearate_data(root_path, input_size, output_size, timestep,foreground_info)

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
