import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
# sys.path.append("/public/home/pfchao/lfpeng/code/time_series_predict/model-main/") 
from CLASS.NN_data_process_0 import data_utils
from CLASS.LOG import make_print_to_file
from CLASS.randomSet import seed_everything
from CLASS.draw import draw_fig
from model.lstm_model import Lstm_Encoder_Decoder
from model.tcn import TemporalConvNet as TCN
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import os

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

# 获得gpu
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'当前设备为{device}')
    return device


# train
def train(net, train_iter, valid_iter, epochs, learning_rate,device,save_path,data_type,model_type):
    train_loss_list = []
    valid_loss_list = []
    bast_loss = np.inf
    # MSE 均方差
    loss_function = nn.MSELoss()  # 定义损失函数
    # MAE 均绝对值误差
    # torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器
    net.to(device)
    loss_function.to(device)
    
    for epoch in range(epochs):
        net.train()
        train_bar = tqdm(train_iter) #使用 tqdm 创建一个进度条，用于追踪训练进度。
        
        train_loss = 0  # 均方误差
        for x_train, decoder_input,y_train in train_bar:
            optimizer.zero_grad()   # 清除之前的梯度
            x_train = x_train.to(device) #[batch_size,num_steps,features_size]      [256,60,51]
            y_train = y_train.to(device) #[256,10,1]
            decoder_input = decoder_input.to(device) #[256,10,1]
            
            # lstm只在训练阶段使用decoder_input
            if model_type == 'LSTM':
                y_train_pred = net(x_train,decoder_input)  # 强制教学
            else:
                y_train_pred = net(x_train)  # 强制教学

            y_train_pred = y_train_pred.to(device)
            # 计算最后一个时间步的预测值的loss，即是否卡顿的loss;注释即计算所有时间步的loss

            loss = loss_function(y_train_pred, y_train).sum()
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            
            train_bar.desc = f'train epoch[{epoch + 1}/{epochs}] loss:{loss}'
            iter_loss = loss.item()
            train_loss += iter_loss
            
        # 计算当前 epoch 的平均损失
        avg_epoch_loss_train = train_loss/len(train_bar)
        train_loss_list.append(avg_epoch_loss_train)    #保存每一个epoch的平均loss    

        #  评估阶段使用验证集valid_iter
        net.eval()
        with torch.no_grad():
            valid_loss = 0 # 均方误差
            for x_valid, decoder_input, y_valid in valid_iter:
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                decoder_input = decoder_input.to(device)
                
                y_valid_pred = net(x_valid)
                y_valid_pred = y_valid_pred.to(device)
                
                # 计算最后一个时间步的预测值的loss，即是否卡顿的loss

                loss_single = loss_function(y_valid_pred, y_valid).sum()
                iter_loss = loss_single.item()
                valid_loss += iter_loss

            # 计算当前 epoch 的平均损失
            avg_epoch_loss_valid = valid_loss/len(valid_iter)
            valid_loss_list.append(avg_epoch_loss_valid)  
            
            print(f'train epoch[{epoch + 1}/{epochs}] avg_loss:{avg_epoch_loss_train} train_all_loss:{train_loss}' )
            print(f'valid epoch[{epoch + 1}/{epochs}] avg_loss:{avg_epoch_loss_valid} valid_all_loss:{valid_loss}' ) 
            print("-------------------------------------------------------") 

            if bast_loss > valid_loss:
                # 训练完毕之后保存模型
                torch.save(model, f'{save_path}/{data_type}_model_{i}.pth')  # 保存模型
                bast_loss = valid_loss

    return net,train_loss_list,valid_loss_list


# predict
def predict(net, test_iter,device):
    print('开始预测')
    # MSE 均方差
    MSEloss_function = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器
    
    net.to(device)
    MSEloss_function.to(device)
    net.eval()
    with torch.no_grad():
        mse_loss = 0 # 均方误差
        mse_loss_list = []
        acc_list,recall_list,precision_list =[],[],[]
        positive_f1_list,negative_f1_list=[],[]
        for x_test, _, y_test in test_iter:
            x_test = x_test.to(device)  #[256,60,51]
            y_test = y_test.to(device)  #[256,10,51]
            
            y_test_pred = net(x_test)  
            y_test_pred = y_test_pred.to(device) #将 y_test_pred 张量移动到指定的设备上
            # 标准化：0-1之间：连续性
            y_test_pred = torch.sigmoid(y_test_pred) #输出的是预测值，经过sigmoid转换一下！！！！
            
            # 查看一下取值范围
            # out=y_test_pred[:,:,[-1]].reshape(-1).float().tolist()
            # print(out)
            # print("is_stuck_np:",max(out),min(out))

            # 这里是将tensor的三维转换为一维
            # [1 if i > 0.5 else 0 for i in is_stuck_pred] 建立threshold，大于0.5为1，小于0.5为0
            is_stuck_pred_0=y_test_pred[:,:,[-1]].reshape(-1)
            is_stuck_pred_1=is_stuck_pred_0>0.5 #后续。。。。。调参，不添加sigmoid
            is_stuck_pred=is_stuck_pred_1.float()
            is_stuck=y_test[:,:,[-1]].reshape(-1)

            # 转换为常量计算F1值
            is_stuck_np=is_stuck.tolist()
            is_stuck_pred_np=is_stuck_pred.tolist()

            positive_f1 = f1_score(is_stuck_np, is_stuck_pred_np, pos_label=0)
            negative_f1 = f1_score(is_stuck_np, is_stuck_pred_np, pos_label=1)

            acc=accuracy_score(is_stuck_np, is_stuck_pred_np)
            recall=recall_score(is_stuck_np, is_stuck_pred_np)
            precision=precision_score(is_stuck_np, is_stuck_pred_np)

            # print("acc:",acc,"recall:",recall,"precision:",precision)
            # F1值
            # print("negative_f1",negative_f1,"positive_f1",positive_f1)

            loss_mse = MSEloss_function(y_test_pred, y_test).sum()
            mse_loss_list.append(float(loss_mse.detach().cpu()))
            mse_loss += loss_mse
               
            positive_f1_list.append(positive_f1)
            negative_f1_list.append(negative_f1)
            acc_list.append(acc)
            recall_list.append(recall)
            precision_list.append(precision)


        # 计算当前 epoch 的平均损失
        avg_mse_loss=  mse_loss / len(test_iter)
        print(f'test avg_mse_loss:{avg_mse_loss}')
        print("acc:",round(sum(acc_list) / len(acc_list),4),"recall:",round(sum(recall_list) / len(recall_list),4),"precision:",round(sum(precision_list) / len(precision_list),4))
        print("卡顿F1：",round(sum(negative_f1_list) / len(negative_f1_list),4),"不卡顿F1：",round(sum(positive_f1_list) / len(positive_f1_list),4))
    return mse_loss_list,positive_f1_list,negative_f1_list


if __name__ == '__main__':
    seed_everything(42)
    # /public/home/pfchao/lfpeng/code/time_series_predict/model-main/Data/wechat/six_diku
    root_path="/opt/data/private/ljx/plf/qos_mi"
    data_type='bj'
    model_type='TCN'   #LSTM or TCN
    data_path=f"{root_path}/Data/BJ_Subway_DataSet"
    save_path=f"{root_path}/output/no_sigmoid/new/{model_type}_ms"
    os.makedirs(save_path, exist_ok=True)
    # loss_figure_name=f"{model_type}_{data_type}_loss"
    
    input_size = 10  # 输入维度
    output_size = 2  # 预测维度
    timestep = 1  # 数据步长
    batch_size = [16,32,64,128,256,512]  # 批量大小
    epochs = 100  # 轮次
    learning_rate = 0.001  # 学习率
    features_size = 36  # 数据特征维度,删除的维度不包括进去；
    scaler_model = StandardScaler()

    # ---------lstm参数----------
    hidden_size = 256  # lstm隐藏层大小 256
    num_layers = 5  # lstm隐藏层个数    5

    type_train = True   #True为训练！！！！！！！！False为预测
    
    # 加载日志
    log_file = make_print_to_file(path='./log')

    #通过data_utils获取数据集：训练、测试、验证
    # (root_path, input_size, output_size, timestep, scaler_model, data_type)
    dataset_train, dataset_valid, dataset_test = data_utils().data_process(
        data_path, input_size, output_size, timestep, scaler_model,data_type)
    # 将数据载入到dataloader
    for i in batch_size:
        print('sigmoid:no',"loss:dataStall--ms")
        print(f'model_type:{model_type}\ndata_type:{data_type}\nscaler_model:{scaler_model}\nbatch_size:{i}\nepochs:{epochs}')
        
        train_loader = DataLoader(dataset=dataset_train, batch_size=i, shuffle=True)
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=i, shuffle=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=i, shuffle=True)
        # 获得设备
        device = get_device()  # 获得设备
        # 获得模型  input_size, hidden_size, num_layers, output_size, batch_size
        if model_type=='LSTM':
            model=Lstm_Encoder_Decoder(features_size, hidden_size, num_layers, output_size)
        elif model_type=='TCN':
            model = TCN(input_size,output_size,features_size)
        
        if type_train:
            # 模型训练net, train_iter, valid_iter, epochs, device, scaler_model
            model,train_loss_list,valid_loss_list = train(model,train_loader,valid_loader,epochs,learning_rate,device,save_path,data_type,model_type)
            # 绘图
            train_loss=pd.Series(train_loss_list)
            valid_loss=pd.Series(valid_loss_list)
            loss_figure_name=f"{model_type}_{data_type}_loss_{i}"
            draw_fig.plot_loss(train_loss,valid_loss,save_path, loss_figure_name)
            
            pd.DataFrame(train_loss_list).to_csv(f'{save_path}/train_loss_list_{i}.csv', index=False, header=False)
            pd.DataFrame(valid_loss_list).to_csv(f'{save_path}/valid_loss_list_{i}.csv', index=False, header=False)

        model = torch.load(f'{save_path}/{data_type}_model_{i}.pth')
        # 模型预测
        mse_loss_list,positive_f1_list,negative_f1_list =predict(model, test_loader,device)
        pd.Series(mse_loss_list).to_csv(f'{save_path}/mse_loss_list_{i}.csv', index=False, header=False)

    log_file.close()