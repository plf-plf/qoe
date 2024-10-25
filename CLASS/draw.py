import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class draw_fig(object):
    #对每一个epoch分别绘制loss
    def subplot_figure_loss(df,epoch,save_path):
        # 读取CSV文件
        # 假设CSV文件中每列都是要绘制的不同的数据序列
        # df = pd.read_csv(path, header=None,names=['Epoch', 'Loss'])

        f = plt.figure(figsize=(12, 7))
        
        for i in range(epoch):
            df_epoch=df[df["Epoch"]==i]
            df_epoch.index = range(1, len(df_epoch) + 1)
            
            # 创建新的图形
            plt.figure(figsize=(8, 4))  # 可以调整图形的大小
            plt.plot(df_epoch.index, df_epoch["Loss"], linestyle='-')
            plt.title(f'Epoch {i+1} Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            f.savefig(f'/public/home/pfchao/lfpeng/code/time_series_predict/ETDataset-main/output/{save_path}/Epoch {i+1} Loss.png', dpi=3000, bbox_inches='tight')

    #对所有epoch分别求均值绘制整体loss
    def plot_figure_loss(df,save_path,data_type):
        # 读取CSV文件
        # 假设CSV文件中每列都是要绘制的不同的数据序列
        # df = pd.read_csv(path, header=None,names=['Epoch', 'Loss'])
        
        f = plt.figure(figsize=(12, 7))
        
        # grouped_mean = df.groupby('Epoch')["Loss"].mean()
        # grouped_mean=grouped_mean.reset_index()
        # grouped_mean.columns=["Epoch","Loss"]
        # 创建新的图形
        plt.figure(figsize=(8, 4))  # 可以调整图形的大小
        plt.title(f'Epoch_Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.plot(df,linestyle='-')
        plt.savefig(f'{save_path}/{data_type}_avg_Loss.png', dpi=3000, bbox_inches='tight')
    
    def plot_loss(train_loss_history, valid_loss_history, save_dir, figureName):
        X = [ i for i in range(1, len(train_loss_history) + 1)]
        Y1 = train_loss_history
        Y2 = valid_loss_history
        plt.figure(figsize=(8, 4))
        plt.plot(X, Y1, label="train_loss")
        plt.plot(X, Y2, label="valid_loss")

        plt.xlabel("epoch")
        plt.ylabel("loss_value")
        plt.legend()
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, figureName))

    
    def model_param(model):
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')
        

if __name__ == "__main__":
    print("这是主程序")