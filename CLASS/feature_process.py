# 特征处理（创造，删除,选择）
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import time
import sys
sys.path.append("../")
import random

class feature_process(object):

    # 删除无用特征
    def drop_feature(self, dataframe):
        drop_list = ['pkg','mLatitude','mLongitude','userScene']
        for f1 in drop_list:
            dataframe = dataframe.drop(f1, axis=1)
        return dataframe
    
    #  带单位的特征处理，【ulRateBps，dlRateBps】：单位为KB/s，将后面的单位部分删除，只保留数值
    def drop_unit(self, dataframe):
        list_deal=['ulRateBps','dlRateBps']
        # dataframe['ulRateBps'] = dataframe['ulRateBps'].replace('KB/s', '').astype(float)
        # dataframe['dlRateBps'] = dataframe['dlRateBps'].replace('KB/s', '').astype(float)
        if 'ulRateBps' in list_deal and 'dlRateBps' in list_deal:
            for f1 in list_deal:
                # 去除单位，只保留数值部分
                dataframe[f1] = dataframe[f1].str.replace('KB/s', '').astype(float)
        else:
            print("列 'ulRateBps' 或 'dlRateBps' 不存在")
        
        return dataframe
    
    # 将True/False值替换为1/0
    def replace_TF(self, df):
        # 将isAirplaneOn的False替换
        df['isAirplaneOn']=df['isAirplaneOn'].replace(['FALSE',False], int(0))
        df['isAirplaneOn']=df['isAirplaneOn'].replace([True, 'TRUE'], int(1))
        # 将label值填充+替换
        df['dataStall'] = df['dataStall'].fillna(int(0))
        df['dataStall'] =  df['dataStall'].replace([True, 'TRUE'], int(1))
        df['dataStall'] =  df['dataStall'].replace(' ', int(0))
        return df
    
    # 特征处理流程整合
    def feature_data_method(self, dataframe):

        # 缺失值填充要在异常值删除之前，因为有一些默认填充值会被判断为异常值。 但是这些填充值被设置未Nan后不会被判断为异常值。
        dataframe = self.drop_feature(dataframe)
        # 带单位的特征处理，删除单位
        dataframe = self.drop_unit(dataframe)
        # 将True/False值替换为1/0
        dataframe = self.replace_TF(dataframe)
        
        return dataframe


    # ！！！！！！！！待补充！！！！！！！！！特殊特征onehot
    def one_hot_rat(self, data):
        """
        :param data: 待onehot数据，一个datamframe
        :return: onehot后的数据，一个datamframe
        """
        rat_list = [-1, 0, 1, 2, 3, 4, 5, 254, 255]
        for i in rat_list:
            data[f'rat_{i}'] = 0
            data[f'rat_{i}'] = data.apply(lambda row: 1 if row['rat'] == i else 0, axis=1)
        data = data.drop('rat', axis=1)
        return data