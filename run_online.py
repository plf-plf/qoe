import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
from CLASS.NN_data_process_online import data_utils
from CLASS.LOG import make_print_to_file
from CLASS.randomSet import seed_everything
from CLASS.draw import draw_fig
from model.lstm_model import Lstm_Encoder_Decoder
from model.tcn import TemporalConvNet as TCN
from TCAN.TCANet_0 import TCANet
from TCAN.BTCN import BTCNet
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import os
import datetime
from CLASS.loss_function import FocalLoss
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

    # 定义损失函数
    # MSE 均方差
    MSEloss_function = nn.MSELoss()  # 定义损失函数
    # loss_function = nn.BCELoss()
    BCEloss_function = nn.BCEWithLogitsLoss()
    focalloss_function = FocalLoss(alpha=0.75, gamma=2)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)  # 定义优化器；一开始所用的优化器
    # optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,weight_decay=0.01)  # 定义优化器
    net.to(device)
    BCEloss_function.to(device)
    MSEloss_function.to(device)
    focalloss_function.to(device)
    loss_bce,loss_mse=0,0
    for epoch in range(epochs):
        net.train()
        train_bar = tqdm(train_iter) #使用 tqdm 创建一个进度条，用于追踪训练进度。
        
        train_loss = 0  # 均方误差
        for x_train, decoder_input,y_train in train_bar:
            optimizer.zero_grad()   # 清除之前的梯度
            x_train = x_train.to(device) #[batch_size,num_steps,features_size]      
            y_train = y_train.to(device) 
            decoder_input = decoder_input.to(device) 
            
            # lstm只在训练阶段使用decoder_input
            if model_type == 'LSTM':
                y_train_pred = net(x_train,decoder_input)  # 强制教学
            else:
                y_train_pred = net(x_train)  # 强制教学

            y_train_pred = y_train_pred.to(device)
            # 预测值经过sigmoid函数处理，将范围变为0-1之间，进行BCEloss计算
            # y_train_pred = torch.sigmoid(y_train_pred)

            # 计算最后一个特征（dataStall）的预测值的loss，即是否卡顿的loss;注释即计算所有时间步的loss
            y_train_pred_bce=y_train_pred[:,:,[-1]] #后续。调参，不添加sigmoid
            y_train_bce=y_train[:,:,[-1]]
            loss_bce = BCEloss_function(y_train_pred_bce, y_train_bce).sum()

            # 用MSEloss_function计算除了最后一个特征（dataStall）的loss
            y_train_pred_mse=y_train_pred[:,:,:-1] #后续。调参，不添加sigmoid
            y_train_mse=y_train[:,:,:-1]
            loss_mse = MSEloss_function(y_train_pred_mse, y_train_mse).sum()

            loss=loss_bce+loss_mse

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
                # y_valid_pred = torch.sigmoid(y_valid_pred)

                # 计算最后一个时间步的预测值的loss，即是否卡顿的BCEloss_function
                y_valid_pred_bce=y_valid_pred[:,:,[-1]]
                y_valid_bce=y_valid[:,:,[-1]]
                loss_bce = BCEloss_function(y_valid_pred_bce, y_valid_bce).sum()

                loss_mse = 0 # 均方误差
                #用MSEloss_function计算除了最后一个特征（dataStall）的loss
                y_valid_pred_mse=y_valid_pred[:,:,:-1] #后续。调参，不添加sigmoid
                y_valid_mse=y_valid[:,:,:-1]
                loss_mse = MSEloss_function(y_valid_pred_mse, y_valid_mse).sum()

                loss_single=loss_bce+loss_mse
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

    return net,train_loss_list,valid_loss_list,optimizer


# predict
def predict(net, test_iter,device):
    print('开始预测')
    # MSE 均方差
    # loss_function = nn.MSELoss()  # 定义损失函数
    # loss_function = nn.BCELoss()#需要输入已经经过 Sigmoid 函数处理的概率值（范围在0到1之间）
    BCEloss_function = nn.BCEWithLogitsLoss()#直接接受未经过 Sigmoid 函数处理的原始 logits，并在内部自动应用 Sigmoid 函数
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器
    
    net.to(device)
    BCEloss_function.to(device)
    net.eval()
    with torch.no_grad():
        mse_loss = 0 # 均方误差
        acc_list,recall_list,precision_list =[],[],[]
        positive_f1_list,negative_f1_list=[],[]
        for x_test, _, y_test in test_iter:
            x_test = x_test.to(device)  
            y_test = y_test.to(device)  
            
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
            is_stuck_pred_1=is_stuck_pred_0>0.5
            is_stuck_pred=is_stuck_pred_1.float()
            is_stuck=y_test[:,:,[-1]].reshape(-1)

            # 转换为常量计算F1值
            is_stuck_np=is_stuck.tolist()
            is_stuck_pred_np=is_stuck_pred.tolist()
            # 计算F1值
            positive_f1 = f1_score(is_stuck_np, is_stuck_pred_np, pos_label=0)
            negative_f1 = f1_score(is_stuck_np, is_stuck_pred_np, pos_label=1)

            acc=accuracy_score(is_stuck_np, is_stuck_pred_np)
            recall=recall_score(is_stuck_np, is_stuck_pred_np)
            precision=precision_score(is_stuck_np, is_stuck_pred_np)

            # f1_manually=2*acc*recall/(acc+recall)
               
            positive_f1_list.append(positive_f1)
            negative_f1_list.append(negative_f1)
            acc_list.append(acc)
            recall_list.append(recall)
            precision_list.append(precision)

        # 计算当前 epoch 的平均结果
        acc=round(sum(acc_list) / len(acc_list),4)
        recall=round(sum(recall_list) / len(recall_list),4)
        precision=round(sum(precision_list) / len(precision_list),4)
        positive_f1=round(sum(positive_f1_list) / len(positive_f1_list),4)
        negative_f1=round(sum(negative_f1_list) / len(negative_f1_list),4)
        print("卡顿F1：",negative_f1,"不卡顿F1：",positive_f1)
        print("acc:",acc,"recall:",recall,"precision:",precision)
        result_df=[negative_f1,positive_f1,acc,recall,precision]
        print("==============================结束预测===============================")
    return result_df,positive_f1_list,negative_f1_list


if __name__ == '__main__':
    seed_everything(42)
    root_path="/opt/data/private/ljx/plf/qos_mi"
    data_type='online'
    model_type='TCAN'   #LSTM or TCN BTCNet
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data_path=f"{root_path}/Data/10.12"     #/result_1021.xlsx
    save_path=f"{root_path}/output/{current_date}/{model_type}_{data_type}"
    os.makedirs(save_path, exist_ok=True)
    
    input_size = 10  # 输入维度
    output_size = 1  # 预测维度
    timestep = 1  # 数据步长
    batch_size = [128]  # 批量大小
    epochs = 300 # 轮次
    learning_rate = 0.001  # 学习率
    features_size = 22  # 数据特征维度,删除的维度不包括进去；
    scaler_model = MinMaxScaler()
    foreground_info="all" # 0表示抖音，1表示虎牙,"all"表示不区分平台
    loss_function=nn.BCEWithLogitsLoss() # 定义标签的损失函数：FocalLoss、nn.BCEWithLogitsLoss || MSELoss是其他特征的损失函数

    # ---------lstm参数----------
    hidden_size = 256  # lstm隐藏层大小 256
    num_layers = 5  # lstm隐藏层个数    5

    type_train = True   #True为训练！！！！！！！！False为预测
    
    # 加载日志
    log_file = make_print_to_file(path=save_path)

    #通过data_utils获取数据集：训练、测试、验证
    # (root_path, input_size, output_size, timestep, scaler_model, data_type)
    dataset_train, dataset_valid, dataset_test = data_utils().data_process(
        data_path, input_size, output_size, timestep, scaler_model,data_type,foreground_info)
    # 将数据载入到dataloader
    result_list,size_list=[],[]
    for i in batch_size:
        print('sigmoid:yes',"loss:all")
        print(loss_function)
        print(f'model_type:{model_type}\ndata_type:{data_type}\nscaler_model:{scaler_model}\nbatch_size:{i}\nepochs:{epochs}\nforeground_info:{foreground_info}')
        
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
        elif model_type=='TCAN':
            model = TCANet(input_size,output_size,features_size)
        elif model_type=='BiTCN':
            model = BTCNet(input_size,output_size,features_size)
        else:
            print("模型类型错误！！！")
        
        if type_train:
            # 模型训练net, train_iter, valid_iter, epochs, device, scaler_model
            model,train_loss_list,valid_loss_list,optimizer = train(model,train_loader,valid_loader,epochs,learning_rate,device,save_path,data_type,model_type)
            # 绘图
            train_loss=pd.Series(train_loss_list)
            valid_loss=pd.Series(valid_loss_list)
            loss_figure_name=f"{model_type}_{data_type}_loss_{i}"
            draw_fig.plot_loss(train_loss,valid_loss,save_path, loss_figure_name)  
            
            # pd.DataFrame(train_loss_list).to_csv(f'{save_path}/train_loss_list_{i}.csv', index=False, header=False)
            # pd.DataFrame(valid_loss_list).to_csv(f'{save_path}/valid_loss_list_{i}.csv', index=False, header=False)

        model = torch.load(f'{save_path}/{data_type}_model_{i}.pth')
        # 模型预测 [negative_f1,positive_f1,acc,recall,precision]
        result_df,positive_f1_list,negative_f1sss_list =predict(model, test_loader,device)
        result_list.append(result_df)
        size_list.append(f'size_{i}')
    # print('optimizer:',optimizer)
    # 将结果保存到csv文件 
    metrics = ['negative_f1','positive_f1','Accuracy', 'Recall', 'Precision']
    all_result = pd.DataFrame(result_list, columns=metrics).T
    all_result.columns = size_list
    all_result.insert(0, f'{model_type}_{data_type}', metrics)
    all_result.to_csv(f'{save_path}/{model_type}_{data_type}_result.csv', index=False)
    print(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_file.close()