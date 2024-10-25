import debugpy

def test():
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass
# test()
import random
import sys
# sys.path.append("/opt/data/private/ljx/plf/time_series") 
import os
import math
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# dataset
import sys
sys.path.append("/opt/data/private/ljx/plf/qos_mi") 
from CLASS.data_process_online import data_process
import json


class my_dataset(Dataset):
    # dataset = my_dataset([encoder_input, decoder_input, label])
    def __iter__(self):
    # 这里返回一个迭代器
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        # 下标来调用数据
        return self.data[0][idx], self.data[1][idx],self.data[2][idx]
    def __len__(self):
        return len(self.data[0])
    def copy(self):
        # 实现复制逻辑 在后面合并数据的时候用到了
        return self.data


#X：多变量
#y:卡顿
# 创建对象

my_data_process = data_process()

class data_utils():
    def read_list(self, folder_path):
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
    def collect_data(self,nested_list, collected_data):
        for element in nested_list:
            if isinstance(element, list):
                if all(not isinstance(i, list) for i in element):
                    collected_data.append(element)
                else:
                    self.collect_data(element,collected_data)
                    
    def get_data_raw(self, folder_path):
        data_list=[]
        # 重新定义特征
        new_columns=["hwLevel","mLastPredictionPci","foreground_info","time","mLastPredictionCid",
                    "rat", "rsrp", "rsrq", "snr", "pci",
                    "deltaTcpKernelLossRate" ,"deltaDnsRttAvg","deltaTcpKernelRttAvg" ,"dlRateBps" ,"ulRateBps",
                    "retans", "totalretrans" ,"unacked" ,"retransmit" ,"lostCount", "avgRttVar" ,"minRtt","avgRtt",
                    "DataStall","isVideoFront"]
        # 读取Excel文件
        all_data_list = self.read_list(folder_path)
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
                self.collect_data(df_1, collected_data) 
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
    def data_file_process_NN(self, Data_list,foreground_info='all'):
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

    # 对于每个数据文件都要变化成为对应的x和label


    def get_dataset(self, data, input_size, output_size, timestep):
        '''
        :param data: 数据集类型（train、test、validation;   data为数据集列表list
        :param input_size:输入的维度，默认为10
        :param output_size:每个样本的预测维度，默认为10
        :param timestep: 时间步，滑动窗口
        decoder_input：只在transformer用
        如果 input_size 是 10，output_size 是 2，并且 timestep 是 1
        那么每个窗口将包含 12 个数据点；其中前 10 个用于模型输入，后 2/3 个用于模型输出。
        窗口之间每次滑动一个数据点。这种方法通常用于准备时间序列数据，以便进行序列到序列的学习任务。
        Return:返回的是一个个序列的集合[[],[],[]]
        '''
        # 将每个文件，分别划分成长度为input_size+out_size的窗口
        data_windows = []
        for df in data:
            for index in range(0, len(df) - input_size - output_size, timestep):
                data_windows.append(df[index:index + input_size + output_size])

        # 未划分decoder encoder输入 target
        data_all = torch.tensor(np.array(data_windows)).to(torch.float32)
        encoder_input = data_all[:, :input_size, :] #从 data_all 中截取前 input_size 个时间步的数据 [x,60,51]
        label = data_all[:, input_size:, :]     #从 data_all 中截取从 input_size 开始到最后的 output_size 个时间步的数据 [x,10,51]

        # 不同transfomer型model，decoder的输入也不同，所以这decoder_input就是随便写的，用于占位。
        start_tgt = torch.zeros((label.shape[0], 1, label.shape[2]))    #[x,1,51]
        decoder_input = torch.cat((start_tgt, label[:, :-1, :]), dim=1) #[x,10,51]
        # 提取编码器输入（encoder_input）和标签（label）;构造解码器输入（decoder_input），通常在Transformer模型中使用
        dataset = my_dataset([encoder_input, decoder_input, label])
        
        return dataset
    
    # 2、合并不同文件中的数据,并进行标准化：这是对训练集数据进行fit，对测试集数据进行transform
    def concat_Data(self, Data, scaler_model):
        '''
        Args:
            Data: 是数据列表list
            scaler_model: 标准化模型
        Returns:
            scaler：使用训练集数据fit拟合scaler
            df_all：标准化后的数据集的列表
        '''
        # 将data合并
        length = [0, len(Data[0])]
        Data_total = Data[0].drop('time', axis=1)
        for i in Data[1:]:
            i = i.drop('time', axis=1)
            Data_total = pd.concat([Data_total, i.copy()])
            length.append(len(i) + length[-1])
        # 合并之后进行标准化
        # 创建MinMaxScaler实例
        scaler = scaler_model
        # 定义哪些特征需要标准化
        columns_to_keep = ['pci',"DataStall","isVideoFront"]
        columns_to_scale = [col for col in Data_total.columns if col not in columns_to_keep]
        # 使用训练集数据fit拟合scaler
        scaler.fit(Data_total[columns_to_scale])
        # 用transform转换训练集数据：
        Data_total[columns_to_scale] = scaler.transform(Data_total[columns_to_scale])
        # 标准化之后在进行拆分
        df_all = []
        for i in range(len(length) - 1):
            df_all.append(Data_total[length[i]:length[i + 1]])
        # 返回包含拆分后数据集的列表
        return scaler, df_all

    # 这里是用训练集的参数scaler来对测试集数据进行transform
    def concat_Data_test(self, Data, scaler_model):
        # 将data合并
        length = [0, len(Data[0])]
        Data_total = Data[0].drop('time', axis=1)
        for i in Data[1:]:
            i = i.drop('time', axis=1)
            Data_total = pd.concat([Data_total, i.copy()])
            length.append(len(i) + length[-1])
        # 合并之后进行标准化
        # 定义哪些特征需要标准化
        columns_to_keep = ['pci',"DataStall","isVideoFront"]
        columns_to_scale = [col for col in Data_total.columns if col not in columns_to_keep]
        # 使用训练集的参数来转换测试集数据,scaler_model是已经用训练集拟合的模型
        Data_total[columns_to_scale] = scaler_model.transform(Data_total[columns_to_scale])
        # 标准化之后在进行拆分
        df_all = []
        for i in range(len(length) - 1):
            # #这里改过了，之前是Data_total[length[i]:length[i + 1]-1]，因为最后一个数据点是没有label的，所以要减1
            df_all.append(Data_total[length[i]:length[i + 1]])
        # 返回包含拆分后数据集的列表
        return df_all

        # 统计数据集中的样本数量，标签数量
    def count_datanum(self,df_all):
        sum=0
        stuck_num=0
        for df in df_all:
            sum+=df.shape[0]
            stuck_num+=df['DataStall'].sum()
        return sum,stuck_num
    
    
    # 加载数据并划分数据，每次调用保证测试集不变，防止信息泄露
    def data_process(self, root_path, input_size, output_size, timestep, scaler_model, data_type,foreground_info):
        '''
        :param root_path: 根目录
        :param input_size:输入的维度，默认为10
        :param output_size:每个样本的预测维度，默认为10
        :param timestep: 时间步，滑动窗口
        '''
        # 获取对应类型的数据,在get_data_raw里面完成了---
        # appname:
        Data_list = self.get_data_raw(root_path)
        print('数据读取结束------------------------------------》')

        # 对每一个文件进行缺失值填充以及特征工程
        Data_list_feature = self.data_file_process_NN(Data_list, foreground_info)
        print('缺失填充及特征工程结束------------------------------------》')

        # 训练集、测试集、验证集的划分比例
        data_list_train = []
        data_list_validation = []
        data_list_test = []
        # 5.8  划分训练集，取所有文件数据（438个文件）的前60% 20% 20%
        num_train = math.ceil(len(Data_list_feature) * 0.6)
        num_validation = math.ceil(len(Data_list_feature) * 0.2)
        num_test = math.ceil(len(Data_list_feature) * 0.2)

        data_list_train = Data_list_feature[0:num_train]
        data_list_validation = Data_list_feature[num_train:num_train + num_validation]
        data_list_test = Data_list_feature[num_train + num_validation:num_train + num_test + num_validation]

        print('训练、验证、测试划分结束------------------------------------》')
        sum_train,stuck_num_train = self.count_datanum(data_list_train)
        sum_valid, stuck_num_valid = self.count_datanum(data_list_validation)
        sum_test, stuck_num_test = self.count_datanum(data_list_test)
        print('训练集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_train,stuck_num_train,f'{round((stuck_num_train/sum_train)*100,2)}%'))
        print('验证集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_valid, stuck_num_valid,f'{round((stuck_num_valid/sum_valid)*100,2)}%'))
        print('测试集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_test, stuck_num_test, f'{round((stuck_num_test/sum_test)*100,2)}%'))


        # concat连接,对数据进行标准化处理
        scaler_X_train, train = self.concat_Data(data_list_train, scaler_model)
        # 使用训练集的参数来转换测试集数据
        test = self.concat_Data_test(data_list_test, scaler_X_train)
        validation = self.concat_Data_test(data_list_validation, scaler_X_train)
        print('无量纲化结束------------------------------------》')

        dataset_train = self.get_dataset(train, input_size, output_size, timestep)
        dataset_validation = self.get_dataset(validation, input_size, output_size, timestep)
        dataset_test = self.get_dataset(test, input_size, output_size, timestep)
        # n=dataset_validation.shape[-1] #这里是统计数量
        return dataset_train, dataset_validation, dataset_test

    # 加载数据并划分数据，每次调用保证测试集不变，防止信息泄露
    def data_process_test(self, root_path, input_size, output_size, timestep, scaler_model, data_type,foreground_info):
        '''
        :param root_path: 根目录
        :param input_size:输入的维度，默认为10
        :param output_size:每个样本的预测维度，默认为10
        :param timestep: 时间步，滑动窗口
        '''
        # 获取对应类型的数据,在get_data_raw里面完成了---
        # appname:
        Data_list = self.get_data_raw(root_path)
        print('数据读取结束------------------------------------》')

        # 对每一个文件进行缺失值填充以及特征工程--这里只用抖音
        Data_list_feature = self.data_file_process_NN(Data_list, 0)
        Data_list_feature_huya=self.data_file_process_NN(Data_list, 1)
        print('缺失填充及特征工程结束------------------------------------》')

        # 训练集、测试集、验证集的划分比例
        data_list_train = []
        data_list_validation = []
        data_list_test = []
        # 5.8  划分训练集，取所有文件数据（438个文件）的前60% 20% 20%
        num_train = math.ceil(len(Data_list_feature) * 0.6)
        num_validation = math.ceil(len(Data_list_feature) * 0.2)
        num_test = math.ceil(len(Data_list_feature) * 0.2)

        data_list_train = Data_list_feature[0:num_train]
        data_list_validation = Data_list_feature[num_train:num_train + num_validation]
        data_list_test = Data_list_feature[num_train + num_validation:num_train + num_test + num_validation]

        data_list_train.extend(Data_list_feature_huya)
        print('训练、验证、测试划分结束------------------------------------》')
        sum_train,stuck_num_train = self.count_datanum(data_list_train)
        sum_valid, stuck_num_valid = self.count_datanum(data_list_validation)
        sum_test, stuck_num_test = self.count_datanum(data_list_test)
        print('训练集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_train,stuck_num_train,f'{round((stuck_num_train/sum_train)*100,2)}%'))
        print('验证集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_valid, stuck_num_valid,f'{round((stuck_num_valid/sum_valid)*100,2)}%'))
        print('测试集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_test, stuck_num_test, f'{round((stuck_num_test/sum_test)*100,2)}%'))


        # concat连接,对数据进行标准化处理
        scaler_X_train, train = self.concat_Data(data_list_train, scaler_model)
        # 使用训练集的参数来转换测试集数据
        test = self.concat_Data_test(data_list_test, scaler_X_train)
        validation = self.concat_Data_test(data_list_validation, scaler_X_train)
        print('无量纲化结束------------------------------------》')

        dataset_train = self.get_dataset(train, input_size, output_size, timestep)
        dataset_validation = self.get_dataset(validation, input_size, output_size, timestep)
        dataset_test = self.get_dataset(test, input_size, output_size, timestep)
        # n=dataset_validation.shape[-1] #这里是统计数量
        return dataset_train, dataset_validation, dataset_test

    # 加载数据并划分数据，每次调用保证测试集不变，防止信息泄露
    def data_process_test_2(self, root_path, input_size, output_size, timestep, scaler_model, data_type,foreground_info):
        '''
        :param root_path: 根目录
        :param input_size:输入的维度，默认为10
        :param output_size:每个样本的预测维度，默认为10
        :param timestep: 时间步，滑动窗口
        '''
        # 获取对应类型的数据,在get_data_raw里面完成了---
        # appname:
        Data_list = self.get_data_raw(root_path)
        print('数据读取结束------------------------------------》')

        # 对每一个文件进行缺失值填充以及特征工程--这里只用抖音
        Data_list_feature = self.data_file_process_NN(Data_list, 0)
        Data_list_feature_huya=self.data_file_process_NN(Data_list, 1)
        print('缺失填充及特征工程结束------------------------------------》')

        # 训练集、测试集、验证集的划分比例
        data_list_train = []
        data_list_validation = []
        data_list_test = []
        #划分训练集、测试集，取所有文件数据（438个文件）的前80% 20%
        num_train = math.ceil(len(Data_list_feature) * 0.8)
        num_test = math.ceil(len(Data_list_feature) * 0.2)

        data_list_train = Data_list_feature[0:num_train]
        data_list_test = Data_list_feature[num_train :num_train + num_test]

        # 加入虎牙，划分训练集、验证集的划分比例
        Data_list_feature_huya.extend(data_list_train)
        Data_list_feature_all=Data_list_feature_huya.copy()
        random.shuffle(Data_list_feature_all)
        num_train = math.ceil(len(Data_list_feature_all) * 0.8)
        num_validation = math.ceil(len(Data_list_feature_all) * 0.2)

        data_list_train = Data_list_feature_all[0:num_train]
        data_list_validation = Data_list_feature_all[num_train:num_validation+num_train]

        print('训练、验证、测试划分结束------------------------------------》')
        sum_train,stuck_num_train = self.count_datanum(data_list_train)
        sum_valid, stuck_num_valid = self.count_datanum(data_list_validation)
        sum_test, stuck_num_test = self.count_datanum(data_list_test)
        print('训练集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_train,stuck_num_train,f'{round((stuck_num_train/sum_train)*100,2)}%'))
        print('验证集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_valid, stuck_num_valid,f'{round((stuck_num_valid/sum_valid)*100,2)}%'))
        print('测试集样本数：{},卡顿样本数为：{},卡顿比例为：{}'.format(sum_test, stuck_num_test, f'{round((stuck_num_test/sum_test)*100,2)}%'))

        # concat连接,对数据进行标准化处理
        scaler_X_train, train = self.concat_Data(data_list_train, scaler_model)
        # 使用训练集的参数来转换测试集数据
        test = self.concat_Data_test(data_list_test, scaler_X_train)
        validation = self.concat_Data_test(data_list_validation, scaler_X_train)
        print('无量纲化结束------------------------------------》')

        dataset_train = self.get_dataset(train, input_size, output_size, timestep)
        dataset_validation = self.get_dataset(validation, input_size, output_size, timestep)
        dataset_test = self.get_dataset(test, input_size, output_size, timestep)
        # n=dataset_validation.shape[-1] #这里是统计数量
        return dataset_train, dataset_validation, dataset_test


if __name__ == '__main__':
    # test()
    path=r'/opt/data/private/ljx/plf/qos_mi/Data/10.12/result.xlsx'
    dfs=data_utils().get_data_raw(path)
    dfs=data_utils().data_file_process_NN(dfs)
    for df in dfs:
        print(df.shape)
