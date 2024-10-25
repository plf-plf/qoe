# 数据处理（缺失值，异常值，无量纲化，数据分析）
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class data_process(object):
    def Missing_value_process_method1(self, dataframe, feature_list):
        """
        :param dataframe:
        :param feature_list: 要进行缺失填充的特征名称列表
        :return:dataframe（缺失值已填充）
        """
        # 循环检查并删除首行缺失值
        while not dataframe.empty and dataframe.iloc[0].isnull().any():
            dataframe = dataframe.drop(index=0).reset_index(drop=True)
        # 其他的使用向下填充bfill
        for feature_name in feature_list:
            dataframe[feature_name] = dataframe[feature_name].fillna(axis=0, method='bfill', inplace=False)
        return dataframe

    # 缺失值处理方法②，对给定的特征使用前一个有效值进行填充。
    def Missing_value_process_method2(self, dataframe, feature_list):
        """
        :param dataframe:
        :param feature_list: 要进行缺失填充的特征名称列表
        :return:dataframe（缺失值已填充）
        """
        # 循环检查并删除首行缺失值
    
        while not dataframe.empty and dataframe.iloc[0].isnull().any():
            dataframe = dataframe.drop(index=0).reset_index(drop=True)
        #其他的使用向上填充ffill
        for feature_name in feature_list:
            dataframe[feature_name] = dataframe[feature_name].fillna(axis=0, method='ffill', inplace=False)
        return dataframe
    
    def datatime_split(self, dataframe):
        """
        :param dataframe:
        :param time_interval: 时间间隔，单位为秒
        :return: dataframe(按时间间隔分割）
        """
        # 按时间间隔分割数据
        dataframe['time'] = pd.to_datetime(dataframe['time'])
        # 计算时间差值
        df_time_diff = dataframe['time'].diff().dt.total_seconds()
        # 初始化一个列表来存储拆分后的 DataFrames
        split_dfs = []
        # 初始化起始索引
        start_idx = 0
        # 遍历 DataFrame，找到时间差值大于 60 秒的断点
        for idx in dataframe[df_time_diff > 60].index:
            split_dfs.append(dataframe.iloc[start_idx:idx].reset_index(drop=True))
            start_idx = idx
            # 添加最后一个分段
            split_dfs.append(dataframe.iloc[start_idx:].reset_index(drop=True))
        split_dfs = [df for df in split_dfs if len(df) >= 12]

        #  # 打印拆分后的 DataFrames
        # for i, df in enumerate(split_dfs):
        #     print(f"DataFrame {i+1}:")
        #     print(df)

        return split_dfs
    



    # 异常值处理方法①，将出现特征异常值的样本直接删除.(添加边界值的版本）
    # 异常值的判断：1、离散数据：特征值不在特征的取值范围内；2、范围数据：特征值不在特征的取值范围内
    # 补充下面代码：这个代码是示范性质的，需要根据实际情况进行修改。
    def abnormal_value_process_method1(self, dataframe):
        """
        :param dataframe:
        :return: dataframe(异常值已删除）
        """
        # 异常值检测:将各个特征的取值按照离散或范围进行分类，然后进行异常值判断。
        # 离散数据
        dict_range_closed = {
            # 'userScene':[1,17],
            'cid':[0,1007],
            'rsrp':[-140,-43],
            'rsrq':[-43,20],
            'snr':[-23,40],
            'ValidLinkTimeRate':[0,100],
            'tcpFwmarkLossRate':[0,100],
            'dnsSuccessRate':[0,100],
            'deltaValidLinkTimeRate':[0,100],
            'deltaTcpFwmarkLossRate':[0,100],
            'deltaTcpKernelLossRate':[0,100],
            'deltaDnsSuccessRate':[0,100],
            'delay':[0,460],
        }
        # 范围数据,closed interval，不会识别nan，因此缺失值需要提前异常值处理，将缺失值变为nan，否则会被视为异常值删除。
        dict_range_opened = {
            'tcpFwmarkRttAvg': [0, np.inf],
            'tcpKernelRttAvg': [0, np.inf],
            'dnsRttAvg': [0, np.inf],
            'deltaTcpFwmarkRttAvg': [0, np.inf],
            'deltaTcpKernelRttAvg': [0, np.inf],
            'deltaDnsRttAvg': [0, np.inf],
            'ulRateBps': [0, np.inf],
            'dlRateBps': [0, np.inf],
            'linkKeepAliveTime': [0, np.inf],
            'avgRtt': [0, np.inf],
            'avgRttVar': [0, np.inf],
            'avgRcvRtt': [0, np.inf],
           'sentCount': [0, np.inf],
           'recvCount': [0, np.inf],
            'lostCount': [0, np.inf],
           'retans': [0, np.inf],
           'retransmit': [0, np.inf],
            'unacked': [0, np.inf],
            'totalretrans': [0, np.inf],
        }
        # 离散数据异常检测
        dict_dispersed={
            # 'userScene':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
            'rat':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
            'dataStall':[0,1],
        }


        condition_all = False

        # 范围数据异常检测
        for key, value in dict_range_closed.items():
            condition = (dataframe[key] < value[0]) + (dataframe[key] > value[1])
            condition_all += condition
        
        # 范围数据,开区间
        for key, value in dict_range_opened.items():
            condition = (dataframe[key] < value[0]) + (dataframe[key] >= value[1])
            condition_all += condition

        # 开始对特征进行判定,没有异常为false，有异常为true
        for key, value in dict_dispersed.items():
            condition = ~dataframe[key].isin(value)  # 不在范围内，为True（异常）
            condition_all += condition
        print("原始数据",dataframe.shape)    
        dataframe = dataframe.drop(dataframe.loc[condition_all].index)
        print("处理后数据",dataframe.shape)    
        return dataframe
    
    # 数据处理流程整合
    def pretreatment_data_method1(self, dataframe):

        # 缺失值填充要在异常值删除之前，因为有一些默认填充值会被判断为异常值。 
        # 但是这些填充值被设置未Nan后不会被判断为异常值。待补充！！！！！！！！！！！
        # dataframe = self.Missing_value_identify(dataframe)

        # 删除每个采样文件内部的重复值
        dataframe = dataframe.drop_duplicates(subset=['time', 'pdcp_tx_rate'], keep='last')
        # 删除异常值，有边界
        dataframe = self.abnormal_value_process_method1(dataframe)

        return dataframe
    

