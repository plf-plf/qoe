# 数据处理（缺失值，异常值，无量纲化，数据分析）
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class data_process(object):

    
    # 删除无用特征
    def drop_feature(self,dataframe, drop_list):
        # drop_list=['hwLevel', 'time']
        for f1 in drop_list:
            dataframe = dataframe.drop(f1, axis=1)
        return dataframe
    
    def foreground_info_regconize(self, dataframe):
        # 场景1：com.ss.android.ugc.aweme:抖音
        # 场景2：com.duowan.kiwi:虎牙
        replace_dict = {"com.ss.android.ugc.aweme": 0, "com.duowan.kiwi": 1}
        # 使用replace方法进行替换
        dataframe["foreground_info"] = dataframe["foreground_info"].replace(replace_dict)
        return dataframe
    
    # 异常值识别
    def abnormal_value_process_method1(self, dataframe):
        """
        :param dataframe:
        :return: dataframe(异常值已删除）
        """
        # 异常值检测:将各个特征的取值按照离散或范围进行分类，然后进行异常值判断。
        # 离散数据
        dict_dispersed = {
            'pci': [0, 1],
            'isVideoFront': [0, 1],
            'rat':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
            'dataStall':[0,1],
        }
        # 范围数据,closed interval，不会识别nan，因此缺失值需要提前异常值处理，将缺失值变为nan，否则会被视为异常值删除。
        dict_range_closed = {
            'mLastPredictionPci': [0, 1007],
            'rsrp': [-140, -43],
            'rsrq': [-43, 20],
            'snr': [-23, 40],
            'rat': [0,25],
            'deltaTcpKernelLossRate': [0, 100],
        }

        dict_range_opened = {
            'deltaTcpKernelRttAvg': [0, np.inf],
            'deltaDnsRttAvg': [0, np.inf],
            'ulRateBps': [0, np.inf],
            'dlRateBps': [0, np.inf],
            'avgRtt': [0, np.inf],
            'avgRttVar': [0, np.inf],
            'minRtt': [0, np.inf],
            'lostCount': [0, np.inf],
            'retans': [0, np.inf],
            'retransmit': [0, np.inf],
            'unacked': [0, np.inf],
            'totalretrans': [0, np.inf],
        }

        condition_all = False

        # 开始对特征进行判定,没有异常为false，有异常为true
        for key, value in dict_dispersed.items():
            condition = ~dataframe[key].isin(value)  # 不在范围内，为True（异常）
            condition_all += condition
        # 范围数据,开区间
        for key, value in dict_range_opened.items():
            condition = (dataframe[key] < value[0]) + (dataframe[key] >= value[1])
            condition_all += condition
        # 范围数据异常检测
        for key, value in dict_range_closed.items():
            condition = (dataframe[key] < value[0]) + (dataframe[key] > value[1])
            condition_all += condition
        dataframe = dataframe.drop(dataframe.loc[condition_all].index)

        return dataframe

    # 异常值识别--这里是将异常值替换为0，然后向上填充
    def abnormal_value_process_method2(self, dataframe):
        """
        :param dataframe:
        :return: dataframe(异常值已删除）
        """
        # 异常值检测:将各个特征的取值按照离散或范围进行分类，然后进行异常值判断。
        # 离散数据
        dict_dispersed = {
            'pci': [0, 1],
            'isVideoFront': [0, 1],
            'rat':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
            'dataStall':[0,1],
        }
        # 范围数据,closed interval，不会识别nan，因此缺失值需要提前异常值处理，将缺失值变为nan，否则会被视为异常值删除。
        dict_range_closed = {
            'mLastPredictionPci': [0, 1007],
            'rsrp': [-140, -43],
            'rsrq': [-43, 20],
            'snr': [-23, 40],
            'rat': [0,25],
            'deltaTcpKernelLossRate': [0, 100],
        }

        dict_range_opened = {
            'deltaTcpKernelRttAvg': [0, np.inf],
            'deltaDnsRttAvg': [0, np.inf],
            'ulRateBps': [0, np.inf],
            'dlRateBps': [0, np.inf],
            'avgRtt': [0, np.inf],
            'avgRttVar': [0, np.inf],
            'minRtt': [0, np.inf],
            'lostCount': [0, np.inf],
            'retans': [0, np.inf],
            'retransmit': [0, np.inf],
            'unacked': [0, np.inf],
            'totalretrans': [0, np.inf],
        }

        condition_all = False

        # 开始对特征进行判定,没有异常为false，有异常为true
        for key, value in dict_dispersed.items():
            condition = ~dataframe[key].isin(value)  # 不在范围内，为True（异常）
            condition_all += condition
        # 范围数据,开区间
        for key, value in dict_range_opened.items():
            condition = (dataframe[key] < value[0]) + (dataframe[key] >= value[1])
            condition_all += condition
        # 范围数据异常检测
        for key, value in dict_range_closed.items():
            condition = (dataframe[key] < value[0]) + (dataframe[key] > value[1])
            condition_all += condition
        dataframe = dataframe.drop(dataframe.loc[condition_all].index)
        
        return dataframe
