# 数据读入，绘制f1等操作
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import math

class data_util(object):

    # 将不同文件数据连接在一起
    def concat_Data(self, Data):
        Data_total = Data[0]
        for i in Data[1:]:
            Data_total = pd.concat([Data_total, i.copy()])

        return Data_total

    # 统计样本数，以及正负样本个数
    def print_data(self, X, Y, type):
        num_X = len(X)
        print(type + "样本数：{}".format(num_X))
        num_positive = Y.sum()
        print(type + "卡顿样本数：{}".format(num_positive))
        print("不卡顿样本：卡顿样本{}".format((num_X - num_positive) / num_positive))

    def split_feature_label_list(self,datalist):
        feature_list,label_list=[],[]
        for df in datalist:
            feature = df.drop('dataStall',axis=1)
            label = df['dataStall']
            feature_list.append(feature)
            label_list.append(label)
        feature_list=self.concat_Data(feature_list)
        label_list=self.concat_Data(label_list) 
        return feature_list,label_list

    def split_feature_label(self,df):
        feature = df.drop('dataStall',axis=1)
        label = df['dataStall']
        return feature,label

    def split_train_test(self, DataList, test_size=0.2):
        data_list_train = []
        data_list_test = []

        for data in DataList:
            num_train = math.ceil(len(data) * (1-test_size))
            num_test = math.ceil(len(data) * test_size)

            train_data = data[0:num_train]
            test_data = data[num_train:num_train + num_test]

            # 将每个data文件划分好训练集之后，分别存放在对应的list里面
            data_list_train.append(train_data)
            data_list_test.append(test_data)

        return data_list_train, data_list_test
    
    def split_train_test_NN(self, Data_list, test_size=0.2):
        data_list_train = []
        data_list_test = []
        data_list_validation = []
        for data in Data_list:

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

        return data_list_train, data_list_test, data_list_validation


    def split_dataframe(self, data, length):
        start = 0
        end = 0
        df_list = []
        for i in range(len(length)):
            end += length[i]
            df = data.iloc[start:end]
            start += length[i]
            df_list.append(df)
        return df_list
    
    def compute_f1(self, model, X_test, y_test):
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        positive_f1_all = []
        negative_f1_all = []
        # y_pred_pro = model.predict_proba(X_test)
        y_pred_pro = model.predict_proba(X_test)
        for i in thresholds:
            #其中大于 i 的被认为是正类（卡顿），小于等于 0.5 的被认为是负类（不卡）,计算的时候i取0.5，即positive_f1_all[8]
            y_test_predictions_high_recall = y_pred_pro[:, 1] > i
            # f1
            positive_f1 = f1_score(y_test, y_test_predictions_high_recall, pos_label=1)
            negative_f1 = f1_score(y_test, y_test_predictions_high_recall, pos_label=0)
            positive_f1_all.append(positive_f1)
            negative_f1_all.append(negative_f1)
        return positive_f1_all, negative_f1_all

    def compute_metrics(self, model, X_test, y_test):
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        y_pred_pro = model.predict_proba(X_test)
        acc_list = []
        recall_list = []
        precision_list = []
        for i in thresholds:
            y_test_predictions_high_recall = y_pred_pro[:, 1] > i
            acc = accuracy_score(y_test_predictions_high_recall,y_test)
            recall = recall_score(y_test_predictions_high_recall,y_test)
            precision = precision_score(y_test_predictions_high_recall,y_test)
            acc_list.append(acc)
            recall_list.append(recall)
            precision_list.append(precision)
        return acc_list, recall_list, precision_list

    def compute_metrics_NN(self, pred, y_test):
        positive_f1_all, negative_f1_all = [], []
        acc_list,recall_list,precision_list = [],[],[]

        positive_f1 = f1_score(y_test, pred, pos_label=1)
        negative_f1 = f1_score(y_test, pred, pos_label=0)
        acc = accuracy_score(pred, y_test)
        recall = recall_score(pred, y_test)
        precision = precision_score(pred, y_test)

        positive_f1_all.append(positive_f1)
        negative_f1_all.append(negative_f1)
        acc_list.append(acc)
        recall_list.append(recall)
        precision_list.append(precision)

        return positive_f1_all, negative_f1_all, acc_list, recall_list, precision_list

if __name__ == "__main__":
    print("这是主程序")