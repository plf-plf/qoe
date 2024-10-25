# 特征相关性分析
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score



class feature_analysis(object):

    def plot_importance(self, columns, importances):
        plt.figure(figsize=(15, 20))
        plt.barh(columns, importances)
        return importances

    # xgb_importance
    def xgb_importance(self, model,X,col_name=None):
        """
        :param model: 已经训练好的xgboost模型
        :param X: 数据特征（dataframe）
        :return: xgboost模型的feature_importance
        """
        feature_importance = model.feature_importances_.tolist()  # 获取模型的特征权重
        columns = X.columns.tolist()  # 获取特征列名称
        plt.figure(figsize=(15, 10))

        # temp_ = pd.Series(feature_importance, index=columns).sort_values(ascending=False)  # 默认为升序，进行降序排列
        temp = pd.Series(feature_importance, index=columns).sort_values()
        plt.barh(temp.index, temp.values)  # 画图展示特征权重
        print(temp.index[::-1])  # 打印特征权重
        plt.show()
        df = pd.DataFrame(temp, index=temp.index, columns=[col_name])
        return df,feature_importance

    def xgb_importance2(self, X, Y, col_name=None):
        model_feature_importance = xgb.XGBClassifier(importance_type='gain')  # 定义模型
        model_feature_importance.fit(X, Y)  # 训练模型
        # 获取模型权重，即每个columns对分类的影响力
        feature_importance = model_feature_importance.feature_importances_.tolist()
        index = X.columns.tolist()  # 获取特征列名称
        temp = pd.Series(feature_importance, index=index).sort_values()#排序，默认降序
        plt.figure(figsize=(10, 8))
        plt.barh(temp.index, temp.values)
        print(temp.index[::-1])
        plt.show()
        df = pd.DataFrame(temp, index=temp.index, columns=[col_name])
        # 返回的是dataframe和权重列表list
        return df, feature_importance


    # permutation importance
    def permutation_importance_mine(self, model, X_test, y_test, random_state=42, n_repeats=40):
        """
        :param model: 定义好的xgboost模型结构
        :param X_test: 测试数据特征
        :param y_test: 测试数据标签
        :param random_state: 随机种子
        :param n_repeats: 打乱次数（越大越耗时，得到的结果也越精准）
        :return:
        """
        y_pred_pro = model.predict_proba(X_test)
        y_test_predictions = y_pred_pro[:, 1] > 0.5  # 以0.5为分类阈值
        f1 = f1_score(y_test, y_test_predictions, pos_label=1)
        baseline_f1 = f1  # 计算基准得分
        print('base{}'.format(baseline_f1))
        importance = []  # 初始化特征重要性
        for col in X_test.columns:
            score_diffs = np.zeros(n_repeats)  # 每个特征要打乱n_repeats次，该变量用于存储每次打乱后的结果
            for j in range(n_repeats):
                num = random_state + j
                rng = np.random.RandomState(num)
                # 打乱第i列特征的顺序
                save = X_test[col].copy()
                X_test[col] = rng.permutation(X_test[col])

                y_permuted_pro = model.predict_proba(X_test)  # 预测的卡顿概率
                y_permuted_pro_predictions = y_permuted_pro[:, 1] > 0.5  # 根据阈值判断是否卡顿
                # 计算f1
                permuted_f1 = f1_score(y_test, y_permuted_pro_predictions, pos_label=1)
                permuted_postive_f1 = f1_score(y_test, y_permuted_pro_predictions, pos_label=0)
                # 计算打乱后的数据的得分差
                permuted_score = permuted_f1
                score_diffs[j] = baseline_f1 - permuted_score

                # 恢复原顺序
                X_test[col] = save

            # 将均值作为最终的permutation importance
            importance.append(np.mean(score_diffs))
        return importance

    # 互信息法
    def mutual_info(self, X, Y, random_state,col_name):
        """
        :param X: 数据特征(dataframe)
        :param Y: 数据标签(dataframe)
        :param random_state: 随机种子(int)一般取42
        :return:每个特征与标签之间的互信息值
        """
        # 从sklearn中导入互信息法接口
        from sklearn.feature_selection import mutual_info_classif as MIC
        result_all = []
        # 由于互信息法具有随机性，因此做5次，然后取平均值
        for i in range(5):
            result = MIC(X, Y, random_state=random_state + i)  # 计算互信息
            result = result.tolist()
            result_all.append(result)
        result_array = np.array(result_all)  # 转换成numpy矩阵格式，便于求平均值
        res = np.mean(result_array, axis=0)  # 求平均值
        res = res.tolist()  # 转换成列表，便于画图
        columns = X.columns.tolist()
        # 将每个特征与标签之间的互信息值绘成柱状图
        # 降序绘图
        plt.figure(figsize=(15, 10))
        temp = pd.Series(res, index=columns).sort_values()  # 排序，默认降序
        plt.barh(temp.index, temp.values)
        # plt.barh(columns, res)

        df = pd.DataFrame(res, index=columns, columns=[col_name])
        # 返回dataframe和互信息列表list
        return df, res

    # spearman相关系数及p值
    def spearman_p(self, X, Y, method='other'):
        """
        用于衡量两个变量之间的单调关系。它通过对两个变量进行等级变换，然后计算等级之间的Pearson相关系数来评估相关性。当两个变量完全单调相关时，系数为+1或-1
        spearman_num_train：衡量两个变量之间的单调关系，取值范围在-1到1之间，绝对值越接近0，越无关
        p值spearman_p_train：评估相关系数的显著性，小 p 值表示相关性显著，大 p 值表示相关性不显著。
        :param X: 数据特征(dataframe)
        :param Y: 数据标签(dataframe)
        :param method: 为'p'表示返回p值，否则返回相关系数
        :return: 每个特征与标签之间的spearman相关系数或p值
        """
        # 导入计算spearman系数的接口
        from scipy import stats
        p_value = []
        corr_s = []
        # 计算每一列特征与标签之间的spearman相关系数、p值。
        # for i in range(len(X.shape[1])):
        for i in range((X.shape[1])):
            # 得到 Spearman 相关系数和 p 值
            corr, pval = stats.spearmanr(X.iloc[:, i], Y)
            p_value.append(pval)
            corr_s.append(corr)
        # 画图展示
        columns = X.columns.tolist()
        plt.figure(figsize=(15, 10))
        # 返回p值或者spearman相关系数
        if method == "p":
            #降序绘图
            temp = pd.Series(p_value, index=columns).sort_values()  # 排序，默认降序
            plt.barh(temp.index, temp.values)
            # plt.barh(columns, p_value)
            # df = pd.DataFrame(p_value, index=columns, columns=[col_name])
            return p_value
        else:
            # 降序绘图
            temp = pd.Series(corr_s, index=columns).sort_values()  # 排序，默认降序
            plt.barh(temp.index, temp.values)
            # df = pd.DataFrame(p_value, index=columns, columns=[col_name])
            # plt.barh(columns, corr_s)
            return corr_s


if __name__ == "__main__":
    print("这是主程序")
