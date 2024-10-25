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
from CLASS.data_process_base import data_process
from CLASS.feature_process import feature_process
# from CLASS.NN_data_split import NN_data_split


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
my_feature_process = feature_process()

class data_utils():
    # 读取数据，对数据进行初步规范
    def get_data_raw(self, path):
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

    # 对每一个文件进行缺失值填充以及特征工程
    def data_file_process_NN(self, Data_list):
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

    # 对于每个数据文件都要变化成为对应的x和label
    # 1、先设置窗口
    def get_dataset_NN_1(self, data, input_size, output_size, timestep):
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
        encoder_input = data_all[:, :input_size, :-1] #从 data_all 中截取前 input_size 个时间步的数据 [x,60,51]
        label = data_all[:, input_size:, [-1]]     #从 data_all 中截取从 input_size 开始到最后的 output_size 个时间步的数据 [x,10,51]

        # 不同transfomer型model，decoder的输入也不同，所以这decoder_input就是随便写的，用于占位。
        start_tgt = torch.zeros((label.shape[0], 1, label.shape[2]))    #[x,1,51]
        decoder_input = torch.cat((start_tgt, label[:, :-1, :]), dim=1) #[x,10,51]
        # 提取编码器输入（encoder_input）和标签（label）;构造解码器输入（decoder_input），通常在Transformer模型中使用
        dataset = my_dataset([encoder_input, decoder_input, label])
        return dataset

    def get_dataset_NN(self, data, input_size, output_size, timestep):
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
      
        data_feature = []
        data_label = []
        seq_lens_train = []
        for df in data:
            for index in range(0, len(df) - input_size-output_size, timestep):
                input_feature = df[index:index + input_size] #这是一个时序的窗口长度
                input_feature = input_feature.drop('dataStall', axis=1)#去掉标签
                data_feature.append(input_feature)
                data_label.append(df[index + input_size:index + input_size + output_size]['dataStall'])#只保留预测的标签
                seq_lens_train.append(input_size)     #长度

        data_feature = torch.tensor(np.array(data_feature)).to(torch.float32)   # 长为20的输入特征，无标签  #[256,20,50]
        data_label = torch.tensor(np.array(data_label)).to(torch.float32)       # 这是第20s的标签          #[256,1]
        # 输入特征、标签和输入序列的长度
        dataset = my_dataset([data_feature,seq_lens_train, data_label])
        return dataset



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
        columns_to_keep = ['dataStall', 'time']
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
        columns_to_keep = ['dataStall', 'time']
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
            stuck_num+=df['dataStall'].sum()
        return sum,stuck_num
    
    
    # 加载数据并划分数据，每次调用保证测试集不变，防止信息泄露
    def data_process(self, root_path, input_size, output_size, timestep, scaler_model, data_type):
        '''
        :param root_path: 根目录
        :param input_size:输入的维度，默认为10
        :param output_size:每个样本的预测维度，默认为10
        :param timestep: 时间步，滑动窗口
        '''
        # 获取对应类型的数据,在get_data_raw里面完成了---
        # appname:
        Data_list,pathName_list = self.get_data_raw(root_path)
        print('数据读取结束------------------------------------》')

        # 对每一个文件进行缺失值填充以及特征工程
        Data_list_feature = self.data_file_process_NN(Data_list)
        print('缺失填充及特征工程结束------------------------------------》')

        # 训练集、测试集、验证集的划分比例
        data_list_train = []
        data_list_test = []
        data_list_validation = []
        # 5.8  划分训练集，取每个文件的前60% 20% 20%
        for data in Data_list_feature:
            num_train = math.ceil(len(data) * 0.6)
            num_test = math.ceil(len(data) * 0.2)
            num_validation = math.ceil(len(data) * 0.2)

            train_data = data[0:num_train]
            test_data = data[num_train:num_train + num_test]
            validation_data = data[num_train + num_test:num_train + num_test + num_validation]

            # 将每个data文件划分好训练集之后，分别存放在对应的list里面
            data_list_train.append(train_data)
            data_list_test.append(test_data)
            data_list_validation.append(validation_data)
        print('训练、验证、测试划分结束------------------------------------》')
        sum_train,stuck_num_train = self.count_datanum(data_list_train)
        sum_valid, stuck_num_valid = self.count_datanum(data_list_validation)
        sum_test, stuck_num_test = self.count_datanum(data_list_test)
        print('训练集样本数：{},卡顿样本数为：{}'.format(sum_train,stuck_num_train))
        print('验证集样本数：{},卡顿样本数为：{}'.format(sum_valid, stuck_num_valid))
        print('测试集样本数：{},卡顿样本数为：{}'.format(sum_test, stuck_num_test))


        # concat连接,对数据进行标准化处理
        scaler_X_train, train = self.concat_Data(data_list_train, scaler_model)
        # 使用训练集的参数来转换测试集数据
        test = self.concat_Data_test(data_list_test, scaler_X_train)
        validation = self.concat_Data_test(data_list_validation, scaler_X_train)
        print('无量纲化结束------------------------------------》')

        dataset_train = self.get_dataset_NN(train, input_size, output_size, timestep)
        dataset_validation = self.get_dataset_NN(validation, input_size, output_size, timestep)
        dataset_test = self.get_dataset_NN(test, input_size, output_size, timestep)
        # n=dataset_validation.shape[-1] #这里是统计数量
        return dataset_train, dataset_validation, dataset_test

if __name__ == '__main__':
    # test()
    path=r'/opt/data/private/ljx/plf/qos_mi/BJ_Subway_DataSet'
    dfs,path_name=data_utils().get_data_raw(path)
    dfs=data_utils().data_file_process_NN(dfs)
    for df in dfs:
        df=data_utils().get_dataset_NN_1(df, 10, 1, 1)
        print(df.shape)
    # for df,name in zip(dfs,path_name):
    #     print(name)
    #     print(df.isna().sum())
    #     # print(df.info())
    #     print(df)
