#coding=utf-8
#这个版本的目的在于从以下四方面提升性能：从数据上提升性能、从算法上提升性能、从算法调优上提升性能、从模型融合上提升性能（性能提升的力度按上表的顺序从上到下依次递减。）
#具体内容可参加https://www.baidu.com/link?url=zdq_sTzndnIZrJL71ZFaLlHnfSblGnNXPzeilgVTaKG2RJEHTWHZHTzVkkipM0El&wd=&eqid=aa03b37b0004b870000000025c2f02e6
#更具体一点地说：可能以后就是增加正则化项吧，能够一定程度的减小网络的复杂度类似奥卡姆剃刀原则。自己随机生成大量的数据吧。将数据缩放到激活函数的阈值内
#原来神经网络模型的训练一直就比较慢，以至于有的时候不一定要采用交叉验证的方式来训练，可能直接用部分未训练数据作为验证集。。
#然后对于模型过拟合或者欠拟合的判断贯穿整个机器学习的过程当中，原来stacking其实是最后一种用于提升模型泛化性能的方式咯。我的面试可以围绕这些开始吧。
#上一个版本的结果不是很理想耶，所以这次真的是最后一次做这个实验了，我理解不应该删除“异常点”，此外随机重采样应该出现了问题了吧，come on, let's do it.
#这个版本和下一个版本综合一起研究了很多的关于如何使用gpu提升计算效率的问题，只有在网络很大且batch-size很大的时候gpu计算速度才能够超过cpu，不论使用tensorflow还是pytorch。
#然后我想到了一个比较奸诈的方式实现计算过程的提速，那就是设置更大的batch-size，毕竟这个参数对于网络的影响还是比较小的但是对于计算时间影响较大的。

#修改内容集被整理如下：
#（0）到这个时候我才发现GPU训练神经网络的速度比cpu训练速度快很多耶。不对呀，好像也没有快很多吧
#现在看来可能是和昨天cpu在运行别的程序有关吧导致计算比较慢，GPU似乎并没有比cpu带来十倍的优势吧？
#所以我觉得可能是我买的台式机被人给坑了吧，不过好在还有GPU可用。就是每次运行之前需要设置device和path咯。
#应该是我的gpu性能太差的缘故，同样价位的gpu性能是同样价位cpu性能的30倍左右吧，所以我现在新买了二手gpu。
#（1）将保存文件的路径修改了。
#（2）特征处理的流程需要修改。尤其是可能增加清除离群点的过程。
#（3）record_best_model_rmse的方式可能需要修改，或许我们需要换种方式获取最佳模型咯，不对好像暂时还不能修改这个东西。
#（4）create_nn_module函数可能需要修改，因为每层都有dropout或者修改为其他结构如回归问题咯。
#（5）noise_augment_dataframe_data可能需要修改，因为Y_train或许也需要增加噪声的。
#（6）nn_f可能需要修改，因为noise_augment_dataframe_data的columns需要修改咯，还有评价准则可能需要优化或者不需要加噪声吧？但是暂时不知如何优化
#（7）nn_stacking_f应该是被弃用了，因为之前我尝试过第二层使用神经网络或者tpot结果都不尽如人意咯，第二层使用逻辑回归才是王道。
#（8）parse_nodes、parse_trials、space、space_nodes需要根据每次的数据修改，best_nodes本身不需要主要是为了快速测试而存在。max_epoch需要根据数据集大小调整。
#（9）train_nn_model、train_nn_model_validate1或许需要换种方式获取最佳模型咯。现在已经找到最佳方式选择模型咯
#（10）nn_stacking_predict应该是被弃用了，因为这个函数是为单模型（节点）开发的预测函数。
#（11）lr_stacking_predict应该是被弃用了，因为这个函数没有超参搜索出最佳的逻辑回归值，计算2000次结果都是一样的。
#（12）tpot_stacking_predict应该是被弃用了，因为第二层使用神经网络或者tpot结果都不尽如人意咯，第二层使用逻辑回归才是王道。
#（13）get_oof回归问题可能需要改写
#（14）train_nn_model、train_nn_model_validate1、train_nn_model_noise_validate2这三系列函数可能需要修改device设置和噪声相关设置。
import os
import sys
import math
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

#原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.feature_extraction import DictVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import KFold, RandomizedSearchCV

import skorch
from skorch import NeuralNetRegressor

from sklearn import svm
from sklearn.covariance import EmpiricalCovariance, MinCovDet

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from tpot import TPOTClassifier

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from mlxtend.classifier import StackingCVClassifier

from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, LassoLarsCV 
from nltk.classify.svm import SvmClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

import seaborn as sns
import matplotlib.pyplot as plt

#load train data and test data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
temp = data_train["SalePrice"]
data_train = data_train.drop(["SalePrice"], axis=1)
data_all = pd.concat([data_train, data_test], axis=0)
#data_all.to_csv("data_all.csv", index=False)

#fill missing numerical feature value by mean value
#fill missing string feature value by mode number or "Null"
#if most of the feature value is missing, then fill them by "Null" 
#otherwise fill missing feature value by mode number
data_all["MSZoning"].fillna(data_all["MSZoning"].mode()[0], inplace=True)
data_all["LotFrontage"].fillna(data_all["LotFrontage"].mean(), inplace=True)
data_all["Alley"].fillna("Null", inplace=True)
data_all["Utilities"].fillna(data_all["Utilities"].mode()[0], inplace=True)
data_all["Exterior1st"].fillna(data_all["Exterior1st"].mode()[0], inplace=True)
data_all["Exterior2nd"].fillna(data_all["Exterior2nd"].mode()[0], inplace=True)
data_all["MasVnrType"].fillna(data_all["MasVnrType"].mode()[0], inplace=True)
data_all["MasVnrArea"].fillna(data_all["MasVnrArea"].mean(), inplace=True)
data_all["BsmtQual"].fillna(data_all["BsmtQual"].mode()[0], inplace=True)
data_all["BsmtCond"].fillna(data_all["BsmtCond"].mode()[0], inplace=True)
data_all["BsmtExposure"].fillna(data_all["BsmtExposure"].mode()[0], inplace=True)
data_all["BsmtFinType1"].fillna(data_all["BsmtFinType1"].mode()[0], inplace=True)
data_all["BsmtFinSF1"].fillna(data_all["BsmtFinSF1"].mean(), inplace=True)
data_all["BsmtFinType2"].fillna(data_all["BsmtFinType2"].mode()[0], inplace=True)
data_all["BsmtFinSF2"].fillna(data_all["BsmtFinSF2"].mean(), inplace=True)
data_all["TotalBsmtSF"].fillna(data_all["TotalBsmtSF"].mean(), inplace=True)
data_all["1stFlrSF"].fillna(data_all["1stFlrSF"].mean(), inplace=True)
data_all["BsmtHalfBath"].fillna(data_all["BsmtHalfBath"].mean(), inplace=True)
data_all["FullBath"].fillna(data_all["FullBath"].mean(), inplace=True)
data_all["KitchenQual"].fillna(data_all["KitchenQual"].mode()[0], inplace=True)
data_all["TotRmsAbvGrd"].fillna(data_all["TotRmsAbvGrd"].mean(), inplace=True)
data_all["Fireplaces"].fillna(data_all["Fireplaces"].mean(), inplace=True)
data_all["GarageType"].fillna(data_all["GarageType"].mode()[0], inplace=True)
data_all["GarageYrBlt"].fillna(data_all["GarageYrBlt"].mean(), inplace=True)
data_all["GarageFinish"].fillna(data_all["GarageFinish"].mode()[0], inplace=True)
data_all["GarageCars"].fillna(data_all["GarageCars"].mean(), inplace=True)
data_all["GarageArea"].fillna(data_all["GarageArea"].mean(), inplace=True)
data_all["GarageQual"].fillna(data_all["GarageQual"].mode()[0], inplace=True)
data_all["GarageCond"].fillna(data_all["GarageCond"].mode()[0], inplace=True)
data_all["PoolQC"].fillna("Null", inplace=True)
data_all["Fence"].fillna("Null", inplace=True)
data_all["FireplaceQu"].fillna(data_all["FireplaceQu"].mode()[0], inplace=True)
data_all["Electrical"].fillna(data_all["Electrical"].mode()[0], inplace=True)
data_all["MiscFeature"].fillna("Null", inplace=True)
data_all["SaleType"].fillna(data_all["SaleType"].mode()[0], inplace=True)
data_all["BsmtUnfSF"].fillna(data_all["BsmtUnfSF"].mean(), inplace=True)
data_all["Functional"].fillna(data_all["Functional"].mode()[0], inplace=True)
data_all["BsmtFullBath"].fillna(data_all["BsmtFullBath"].mean(), inplace=True)

#add sales prices feature to training data
data_train = pd.concat([data_train, temp], axis=1)
#data_all.to_csv("C:\\Users\\win7\\Desktop\\temp1.csv", index=False)
 
#use DictVectorizer to create something like one-hot encoding
dict_vector = DictVectorizer(sparse=False)
#delete Id feature from all data to avoid negative effect on prediction
data_all = data_all.drop(["Id"], axis=1)
X_all = data_all
X_all = dict_vector.fit_transform(X_all.to_dict(orient='record'))
X_all = pd.DataFrame(data=X_all, columns=dict_vector.feature_names_)
#scale feature to (0,1), which could make process of training easier
X_all_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_all), columns = X_all.columns)
X_all_scaled = pd.DataFrame(data = X_all_scaled, index = X_all.index, columns = X_all_scaled.columns.values)
X_train_scaled = X_all_scaled[:len(data_train)]
X_test_scaled = X_all_scaled[len(data_train):]
Y_train = data_train["SalePrice"]

#create RMSELoss from pytorch class to train model 
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        
    def forward(self, pred, truth):
        return torch.sqrt(torch.mean((pred-truth)**2))

#determine if two numbers are close, a and b are the numbers
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)    

#calculate the rmse of prediction, Y_train_pred is prediction and Y_train is truth
def cal_rmse(Y_train_pred, Y_train):
    
    #当遇到nan的时候所有的数据都会变成nan
    error = Y_train_pred- Y_train
    error = torch.from_numpy(error)
    return float(torch.sqrt(torch.mean(error*error)))

#calculate the rmsle of prediction, Y_train_pred is prediction and Y_train is truth
def cal_rmsle(Y_train_pred, Y_train):

    Y_train = torch.from_numpy(Y_train)
    Y_train_pred = torch.from_numpy(Y_train_pred)
    error = torch.log(Y_train_pred).double() - torch.log(Y_train).double()
    return float(torch.sqrt(torch.mean(error*error)))
    
#calculate the rmse of prediction of classifier, 
#rsg is classifier Y_train_pred is prediction and Y_train is truth
def cal_nnrsg_rmse(rsg, X_train, Y_train):
    
    Y_train_pred = rsg.predict(X_train.astype(np.float32))
    return cal_rmse(Y_train_pred, Y_train)

#calculate the rmsle of prediction of classifier, 
#rsg is classifier Y_train_pred is prediction and Y_train is truth
def cal_nnrsg_rmsle(rsg, X_train, Y_train):
    
    Y_train_pred = rsg.predict(X_train.astype(np.float32))
    return cal_rmsle(Y_train_pred, Y_train)

#determine whether the file named title_best_model.pickle exists, title is part name of the file
def exist_files(title):
    
    return os.path.exists(title+"_best_model.pickle")
    
#save the intermediate parameter of the bayesian hyperparameters optimization,
#trials is the reocrd of bayesian hyperparameters optimization,
#space_nodes is the optimization space of bayesian hyperparameters, 
#which mainly used to and makes it easier to get the best hyperparameters
#title is the part of file name which above intermediate parameter will be saved
def save_inter_params(trials, space_nodes, best_nodes, title):
 
    files = open(str(title+"_intermediate_parameters.pickle"), "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

#load the intermediate parameter of the bayesian hyperparameters optimization,
#title is the part of file name which the intermediate parameter will be load from
def load_inter_params(title):
  
    files = open(str(title+"_intermediate_parameters.pickle"), "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes

#save the result of dataset stacking,
#stacked_train is stacked train data,
#stacked_test is stacked test data,
#title is the part of file name which above intermediate parameter will be saved
def save_stacked_dataset(stacked_train, stacked_test, title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "wb")
    pickle.dump([stacked_train, stacked_test], files)
    files.close()

#load the result of dataset stacking,
#title is the part of file name which the dataset stacking will be load from
def load_stacked_dataset(title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "rb")
    stacked_train, stacked_test = pickle.load(files)
    files.close()
    
    return stacked_train, stacked_test

#save the best model of the experiment,
#best_model is model to be saved,
#title is the part of file name which above intermediate parameter will be saved
def save_best_model(best_model, title):
    
    files = open(str(title+"_best_model.pickle"), "wb")
    pickle.dump(best_model, files)
    files.close()

#load the best model of the experiment,
#title_and_nodes is the part of file name which the best model will be load from
def load_best_model(title_and_nodes):
    
    files = open(str(title_and_nodes+"_best_model.pickle"), "rb")
    best_model = pickle.load(files)
    files.close()
    
    return best_model

#record the best accuracy of the model in the experiment,
#clf is the newly trained model,
#acc is the accuracy of the clf,
#best_model is the best model so far,
#best_acc is the accuracy of the best_model
def record_best_model_rmse(rsg, rsme, best_model, best_rmse):
    
    flag = False
    
    if not isclose(best_rmse, rsme):
        if best_rmse > rsme:
            flag = True
            best_rmse = rsme
            best_model = rsg
            
    return best_model, best_rmse, flag

#create model in the way of nn.Sequential of pytorch,
#input_nodes is the number of input nodes,
#hidden_layers is the number of hidden layers,
#hidden_nodes is the number of hidden nodes, 
#output_nodes is the number of output nodes,
#percentage is the percentage of dropout
def create_nn_module(input_nodes, hidden_layers, hidden_nodes, output_nodes, percentage=0.1):
    
    module_list = []
    
    #当没有隐藏节点的时候
    if(hidden_layers==0):
        module_list.append(nn.Linear(input_nodes, output_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        #这边softmax的值域刚好就是(0,1)算是符合softmax的值域吧。
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
        
    #当存在隐藏节点的时候
    else :
        module_list.append(nn.Linear(input_nodes, hidden_nodes))
        module_list.append(nn.Dropout(percentage))
        module_list.append(nn.ReLU())
        
        for i in range(0, hidden_layers):
            module_list.append(nn.Linear(hidden_nodes, hidden_nodes))
            module_list.append(nn.Dropout(percentage))
            module_list.append(nn.ReLU())
             
        module_list.append(nn.Linear(hidden_nodes, output_nodes))
            
    model = nn.Sequential()
    for i in range(0, len(module_list)):
        model.add_module(str(i+1), module_list[i])
    
    return model

#init model in different ways,
#rsg is the model to be inited,
#weiht_mode is the ways to init,
#bias is the value to init bias,
#not all weight_mode need the value of bias,
#the ways to init weight in order are: use default weight and bias, 
#use xavier_normal_ to init weight and default bias,
#use xavier_normal_ to init weight and bias,
#use xavier_normal_ to init weight and use the value of the bias to constant init bias,
#use xavier_uniform_ to init weight and default bias,
#use xavier_uniform_ to init weight and bias,
#use xavier_uniform_ to init weight and use the value of the bias to constant init bias,
#use kaiming_normal_ to init weight and default bias,
#use kaiming_normal_ to init weight and bias,
#use kaiming_normal_ to init weight and use the value of the bias to constant init bias,
#use kaiming_uniform_ to init weight and default bias,
#use kaiming_uniform_ to init weight and bias,
#use kaiming_uniform_ to init weight and use the value of the bias to constant init bias,
def init_module(rsg, weight_mode, bias):

    for layer in rsg.modules():
        #我本来想单独处理nn.Linear nn.Conv2d等
        #但是处理的流程好像都是一样的，所以不用分开处理
        #不对啊，nn.Conv2d
        #目前这个可能会遇到dropout relu之类的
        if isinstance(layer, nn.Linear):
            if (weight_mode==1):
                pass
                #weight和bias均使用pytorch默认的初始值，也就是下面的赋值方式咯
                #def reset_parameters(self):
                #stdv = 1. / math.sqrt(self.weight.size(1))
                #self.weight.data.uniform_(-stdv, stdv)
                #if self.bias is not None:
                #    self.bias.data.uniform_(-stdv, stdv)

            #这下面是xavier_normal_的方式初始化的三种情况
            elif (weight_mode==2):
                #使用xavier_normal_的方式初始化weight但是不改变默认bias
                #可能两个函数的std存在差异，不知道是否能够交融在一起
                #现在需要注意的是每个函数需要注意参数，使用默认参数可能造成weight和bias范围不搭
                #uniform_的默认stdv值是stdv = 1. / math.sqrt(self.weight.size(1))
                #xavier_normal_的默认std值是std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                #还好这两个参数的范围是搭的所以可以直接放在一起吧
                nn.init.xavier_normal_(layer.weight.data)
                
            elif (weight_mode==3):
                #使用xavier_normal_的方式初始化weight和bias
                nn.init.xavier_normal_(layer.weight.data)
                #nn.init.xavier_normal_(layer.bias.data)
                #这下面是模拟上面xavier_normal_的初始化方式
                #特别注意了一下bias的std范围和xavier_normal_的是搭配的
                std = 1.0 * math.sqrt(1.0 / layer.weight.data.size(0))
                #这里不计算std的话，可能参数的范围和上面是不搭的
                nn.init.normal_(layer.bias.data, 0, std)
                
            elif (weight_mode==4):
                #使用xavier_normal_的方式初始化weight
                #将bias的值设置为固定的值咯，这样一来好像需要设定很多bias候选值？
                #感觉就提供五到十个选择吧，我理解bias区间范围还是比较窄的（从默认赋值看出）
                nn.init.xavier_normal_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)
                
            #接下来是xavier_uniform_初始化的三类情况
            elif (weight_mode==5):
                #xavier_uniform_的std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                #std = math.sqrt(3.0) * std
                #这个std的默认范围和默认的bias的空间是不搭的，但是还是作为一种初始化方案吧
                nn.init.xavier_uniform_(layer.weight.data)
            
            elif (weight_mode==6):
                nn.init.xavier_uniform_(layer.weight.data)
                #nn.init.xavier_uniform_(layer.bias.data)
                #这下面是模拟上面xavier_uniform_的初始化方式
                std = 1.0 * math.sqrt(1.0 / layer.weight.data.size(0))
                std = math.sqrt(3.0) * std
                #这里不计算std的话，可能参数的范围和上面是不搭的
                nn.init.uniform_(layer.bias.data, -std, std)
                
            elif (weight_mode==7):
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)
            
            #接下来是kaiming_normal_初始化的三类情况
            elif (weight_mode==8):
                #使用kaiming_normal_的方式初始化weight但是不改变默认bias
                nn.init.kaiming_normal_(layer.weight.data)
                
            elif (weight_mode==9):
                nn.init.kaiming_normal_(layer.weight.data)
                gain = math.sqrt(2.0 / (1 + 0** 2))
                fan = layer.bias.data.size(0)
                std = gain / fan
                nn.init.normal_(layer.bias.data, 0, std)
            
            elif (weight_mode==10):
                nn.init.kaiming_normal_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)
                
            #接下来是kaiming_uniform_初始化的三类情况
            elif (weight_mode==11):
                #使用kaiming_uniform_的方式初始化weight但是不改变默认bias
                nn.init.kaiming_uniform_(layer.weight.data)
                
            elif (weight_mode==12):
                nn.init.kaiming_uniform_(layer.weight.data)
                gain = math.sqrt(2.0 / (1 + 0** 2))
                fan = layer.bias.data.size(0)
                std = gain / math.sqrt(fan)
                std = math.sqrt(3.0) * std
                nn.init.uniform_(layer.bias.data, -std, std)
            
            elif (weight_mode==13):
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, bias)                
                
            else:
                pass

#add noise to do data augmentation for dataframe data,
#mean is the mean of the Gaussian noise,
#std is the variance of the Gaussian noise,
#X_train is the data to be added noise,
#columns is the columns of X_train to be added noise
def noise_augment_dataframe_data(mean, std, X_train, Y_train, columns):
    
    X_noise_train = X_train.copy()
    X_noise_train.is_copy = False
    
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train.iloc[i,[j]] += random.gauss(mean, std)

    return X_noise_train, Y_train

#add noise to do data augmentation for ndarray data,
#mean is the mean of the Gaussian noise,
#std is the variance of the Gaussian noise,
#X_train is the data to be added noise,
#columns is the columns of X_train to be added noise
def noise_augment_ndarray_data(mean, std, X_train, Y_train, columns):
    
    X_noise_train = X_train.copy()
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train[i][j] +=  random.gauss(mean, std)
    
    return X_noise_train, Y_train

#the objective function and bayesian hyperparameters optimization will get the minimum value of which,
#params is the current parameters of bayesian hyperparameters optimization.
#which you can learn their usage scenarios from example code in the final of this file.
def nn_f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    print("optimizer__betas", params["optimizer__betas"])
    print("bias", params["bias"])
    print("weight_mode", params["weight_mode"])
    print("patience", params["patience"])
    print("input_nodes", params["input_nodes"])
    print("hidden_layers", params["hidden_layers"])
    print("hidden_nodes", params["hidden_nodes"])
    print("output_nodes", params["output_nodes"])
    print("percentage", params["percentage"])
    
    #我真的觉得这个做法很古怪，但是还是试一下吧，我的实践表明
    rmse_list = []
    rmsle_list = []
    for i in range(0, 5):
        #这边的rsg需要由NeuralNetClassifier修改为NeuralNetRegressor类型
        rsg = NeuralNetRegressor( lr = params["lr"],
                                  optimizer__weight_decay = params["optimizer__weight_decay"],
                                  criterion = params["criterion"],
                                  batch_size = params["batch_size"],
                                  optimizer__betas = params["optimizer__betas"],
                                  module = create_nn_module(params["input_nodes"], params["hidden_layers"], 
                                                            params["hidden_nodes"], params["output_nodes"], params["percentage"]),
                                  max_epochs = params["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=params["patience"])],
                                  device = params["device"],
                                  optimizer = params["optimizer"]
                                  )
        init_module(rsg.module, params["weight_mode"], params["bias"])
        
        #Y_temp = Y_split_train.values.reshape(Y_split_train.shape[0])
        #rsg.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.float32))
        #你妈卖批的，我是真的不知道为什么一定要修改成下面的这个样子才能运行呢？
        #现在应该有个更简单的办法实现这个部分，就是在split的时候就修改咯
        #Y_temp = Y_split_train.values.reshape(-1,1)
        #rsg.fit(X_split_train.values.astype(np.float32), Y_temp.astype(np.float32))
        rsg.fit(X_split_train.values.astype(np.float32), Y_split_train.values.astype(np.float32))
        
        Y_pred = rsg.predict(X_split_test.values.astype(np.float32))
        rmse = cal_rmse(Y_pred, Y_split_test.values)  
        rmse_list.append(rmse)
        rmsle = cal_rmsle(Y_pred, Y_split_test.values)
        rmsle_list.append(rmsle)
    
    rmse_sum = 0.0
    rmsle_sum = 0.0
    for i in range(0, len(rmse_list)):
        #如果存在nan的时候需要将其进行替换否则无法排序hyperopt报错= =！
        #当rmse_list(本身类型为float类型的变量)，下面的判断成立的时候rmse_list为nan
        #只有极少数的情况下才会启用下面的判断咯
        if(rmse_list[i] != rmse_list[i]):
            rmse_list[i] = 99999999999.9
        rmse_sum += rmse_list[i]
        
        if(rmsle_list[i] != rmsle_list[i]):
            rmsle_list[i] = 99999999999.9
        rmsle_sum += rmsle_list[i]
                
    metric1 = rmse_sum/(len(rmse_list))
    metric2 = rmsle_sum/(len(rmsle_list))
     
    print(metric1)
    print(metric2)
    print()    
    #回归问题时，这里的方差应该越小越好吧
    #分类问题时，这里的准确率应该越大越好吧
    return metric1

#parse best hyperparameters in the bayesian hyperparameters optimization,
#trials is the record of bayesian hyperparameters optimization,
#space_nodes is the optimization space of bayesian hyperparameters
def parse_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[0]["misc"]["vals"]["max_epochs"][0]]

    best_nodes["lr"] = space_nodes["lr"][trials_list[0]["misc"]["vals"]["lr"][0]] 
    best_nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
    best_nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]]
    best_nodes["weight_mode"] = space_nodes["weight_mode"][trials_list[0]["misc"]["vals"]["weight_mode"][0]]
    best_nodes["bias"] = space_nodes["bias"][trials_list[0]["misc"]["vals"]["bias"][0]]
    best_nodes["patience"] = space_nodes["patience"][trials_list[0]["misc"]["vals"]["patience"][0]]
    best_nodes["device"] = space_nodes["device"][trials_list[0]["misc"]["vals"]["device"][0]]
    best_nodes["optimizer"] = space_nodes["optimizer"][trials_list[0]["misc"]["vals"]["optimizer"][0]]
    
    #新添加的这些元素用于控制模型的结构
    best_nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[0]["misc"]["vals"]["input_nodes"][0]]
    best_nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[0]["misc"]["vals"]["hidden_layers"][0]]
    best_nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[0]["misc"]["vals"]["hidden_nodes"][0]]
    best_nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[0]["misc"]["vals"]["output_nodes"][0]]
    best_nodes["percentage"] = space_nodes["percentage"][trials_list[0]["misc"]["vals"]["percentage"][0]]

    return best_nodes

#parse numbers of best hyperparameters in the bayesian hyperparameters optimization,
#trials is the record of bayesian hyperparameters optimization,
#space_nodes is the optimization space of bayesian hyperparameters, 
#which mainly used to and makes it easier to get the best hyperparameters
#num is the numbers of best hyperparameters to get
def parse_trials(trials, space_nodes, num):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    #nodes = {}nodes如果在外面那么每次更新之后都是一样的咯
    nodes_list = []
    
    for i in range(0, num):
        nodes = {}
        nodes["title"] = space_nodes["title"][trials_list[i]["misc"]["vals"]["title"][0]]
        nodes["path"] = space_nodes["path"][trials_list[i]["misc"]["vals"]["path"][0]]
        nodes["mean"] = space_nodes["mean"][trials_list[i]["misc"]["vals"]["mean"][0]]
        nodes["std"] = space_nodes["std"][trials_list[i]["misc"]["vals"]["std"][0]]
        nodes["batch_size"] = space_nodes["batch_size"][trials_list[i]["misc"]["vals"]["batch_size"][0]]
        nodes["criterion"] = space_nodes["criterion"][trials_list[i]["misc"]["vals"]["criterion"][0]]
        nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[i]["misc"]["vals"]["max_epochs"][0]]
        nodes["lr"] = space_nodes["lr"][trials_list[i]["misc"]["vals"]["lr"][0]] 
        nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[i]["misc"]["vals"]["optimizer__betas"][0]]
        nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[i]["misc"]["vals"]["optimizer__weight_decay"][0]]
        nodes["weight_mode"] = space_nodes["weight_mode"][trials_list[i]["misc"]["vals"]["weight_mode"][0]]
        nodes["bias"] = space_nodes["bias"][trials_list[i]["misc"]["vals"]["bias"][0]]
        nodes["patience"] = space_nodes["patience"][trials_list[i]["misc"]["vals"]["patience"][0]]
        nodes["device"] = space_nodes["device"][trials_list[i]["misc"]["vals"]["device"][0]]
        nodes["optimizer"] = space_nodes["optimizer"][trials_list[i]["misc"]["vals"]["optimizer"][0]]
        nodes["input_nodes"] = space_nodes["input_nodes"][trials_list[i]["misc"]["vals"]["input_nodes"][0]]
        nodes["hidden_layers"] = space_nodes["hidden_layers"][trials_list[i]["misc"]["vals"]["hidden_layers"][0]]
        nodes["hidden_nodes"] = space_nodes["hidden_nodes"][trials_list[i]["misc"]["vals"]["hidden_nodes"][0]]
        nodes["output_nodes"] = space_nodes["output_nodes"][trials_list[i]["misc"]["vals"]["output_nodes"][0]]
        nodes["percentage"] = space_nodes["percentage"][trials_list[i]["misc"]["vals"]["percentage"][0]]
        
        nodes_list.append(nodes)
    return nodes_list

#这个选择最佳模型的时候存在过拟合的风险
#the following 7 functions are different ways to train neural networks,
#nodes is the best hyperparameters for neural networks,
#X_train_scaled is the train data after feature scale,
#max_evals is the number of training,
#I recommend you use train_nn_model_validate1 or train_nn_model_validate2
def train_nn_model(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #由于神经网络模型初始化、dropout等的问题导致网络不够稳定
    #解决这个问题的办法就是多重复计算几次，选择其中靠谱的模型
    best_rmse= 99999999999.9
    best_model = 0.0
    for j in range(0, max_evals):
        
        rsg = NeuralNetRegressor( lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
        rsg.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.float32))
            
        metric = cal_nnrsg_rmse(rsg, X_train_scaled, Y_train)
        print(metric)
        
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)        
    
    return best_model, best_rmse

def train_nn_model_validate1(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #我觉得0.12的设置有点多了，还有很多数据没用到呢，感觉这样子设置应该会好一些的吧？
    #X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.12, stratify=Y_train)
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.14)
    #由于神经网络模型初始化、dropout等的问题导致网络不够稳定
    #解决这个问题的办法就是多重复计算几次，选择其中靠谱的模型
    best_rmse = 99999999999.9
    best_model = 0.0
    for j in range(0, max_evals):
        
        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
        rsg.fit(X_split_train.astype(np.float32), Y_split_train.astype(np.float32))
            
        #Y_pred = rsg.predict(X_split_test.astype(np.float32))
        metric = cal_nnrsg_rmse(rsg, X_split_test, Y_split_test)
        
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)        
    
    return best_model, best_rmse

#这里面采用cross_val_score的方式应该更能够体现出泛化的性能吧。
#这样的交叉验证才是最高效率的利用数据的方式吧。
def train_nn_model_validate2(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #解决这个问题主要还是要靠cross_val_score这样才能够显示泛化性能吧。
    best_rmse= 99999999999.9
    best_model = 0.0
    for j in range(0, max_evals):
        
        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
        
        #这里好像是无法使用skf的呀，不对只是新的skf需要其他设置啊，需要修改Y_train的shape咯
        #skf = StratifiedKFold(Y_train, n_folds=5, shuffle=True, random_state=None)
        #这里sklearn的均方误差是可以为负数的，我还以为是自己的代码出现了问题了呢
        metric = cross_val_score(rsg, X_train_scaled.astype(np.float32), Y_train.astype(np.float32), cv=5, scoring="neg_mean_squared_log_error").mean()
        print(metric)
        
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.float32))
    return best_model, best_rmse

def train_nn_model_noise_validate1(nodes, X_train_scaled, Y_train, max_evals=10):
    
    #我觉得0.12的设置有点多了，还有很多数据没用到呢，感觉这样子设置应该会好一些的吧？
    #X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.12, stratify=Y_train)
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.05)
    #由于神经网络模型初始化、dropout等的问题导致网络不够稳定
    #解决这个问题的办法就是多重复计算几次，选择其中靠谱的模型
    best_rmse = 99999999999.9
    best_model = 0.0
    for j in range(0, max_evals):
        
        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
        rsg.fit(X_split_train.astype(np.float32), Y_split_train.astype(np.float32))
            
        metric = cal_nnrsg_rmse(rsg, X_split_test, Y_split_test)
        print(metric)
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)        
    
    return best_model, best_rmse

#然后在这里增加一次噪声和验证咯，感觉我把程序弄的真的好复杂呀？
#或许我下一阶段的实验就是查看是否nn_f不加入噪声只是第二阶段增加噪声效果是否更好？
def train_nn_model_noise_validate2(nodes, X_train_scaled, Y_train, max_evals=10):
    
    X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.1)

    best_rmse = 99999999999.9
    best_model = 0.0
    for j in range(0, max_evals):
        
        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
        
        X_noise_train, Y_noise_train = noise_augment_ndarray_data(nodes["mean"], nodes["std"], X_split_train, Y_split_train, columns=[])#columns=[i for i in range(1, 19)])
        
        rsg.fit(X_noise_train.astype(np.float32), Y_noise_train.astype(np.float32))
            
        metric = cal_nnrsg_rmse(rsg, X_split_test, Y_split_test)
        print(metric)
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)        
    
    return best_model, best_rmse

def train_nn_model_noise_validate3(nodes, X_train_scaled, Y_train, max_evals=10):
    
    best_rmse = 99999999999.9
    best_model = 0.0
    
    #这一轮就使用这一份加噪声的数据就可以了吧？没有必要在下面的for循环中也添加吧？
    #我好像真的只有用这种方式增加stacking模型之间的差异了吧？以提升泛化性能咯。
    X_noise_train, Y_noise_train = noise_augment_ndarray_data(nodes["mean"], nodes["std"], X_train_scaled, Y_train, columns=[])#columns=[i for i in range(0, 19)])

    for j in range(0, max_evals):
        
        #不对吧我在想是不是在这里面添加噪声更好一些呢，毕竟上面的噪声添加方式可能造成模型过渡拟合增加噪声之后的数据？？
        #我不知道是不是在这里面增加噪声得到的效果会更好一些呢，我觉得很郁闷问题到底出现在哪里呀？
        
        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
                
        metric = cross_val_score(rsg, X_noise_train.astype(np.float32), Y_noise_train.astype(np.float32), cv=5, scoring="mean_squared_log_error").mean()
        print(metric)
        
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.float32))
    return best_model, best_rmse

def train_nn_model_noise_validate4(nodes, X_train_scaled, Y_train, max_evals=10):
    
    best_rmse = 99999999999.9
    best_model = 0.0
    
    for j in range(0, max_evals):
        #不对吧我在想是不是在这里面添加噪声更好一些呢，毕竟上面的噪声添加方式可能造成模型过渡拟合增加噪声之后的数据？？
        #我不知道是不是在这里面增加噪声得到的效果会更好一些呢，我觉得很郁闷问题到底出现在哪里呀？
        X_noise_train, Y_noise_train = noise_augment_ndarray_data(nodes["mean"], nodes["std"], X_train_scaled, Y_train, columns = [])#columns=[i for i in range(0, 19)])

        rsg = NeuralNetRegressor(lr = nodes["lr"],
                                  optimizer__weight_decay = nodes["optimizer__weight_decay"],
                                  criterion = nodes["criterion"],
                                  batch_size = nodes["batch_size"],
                                  optimizer__betas = nodes["optimizer__betas"],
                                  module = create_nn_module(nodes["input_nodes"], nodes["hidden_layers"], 
                                                         nodes["hidden_nodes"], nodes["output_nodes"], nodes["percentage"]),
                                  max_epochs = nodes["max_epochs"],
                                  callbacks=[skorch.callbacks.EarlyStopping(patience=nodes["patience"])],
                                  device = nodes["device"],
                                  optimizer = nodes["optimizer"]
                                  )
        init_module(rsg.module, nodes["weight_mode"], nodes["bias"])
                
        #这边的折数由5折修改为10折吧，这样子的话应该更加能够表示出稳定性吧
        #skf = StratifiedKFold(Y_noise_train, n_folds=10, shuffle=True, random_state=None)
        metric = cross_val_score(rsg, X_noise_train.astype(np.float32), Y_noise_train.astype(np.float32), scoring="mean_squared_log_error").mean()
        print(metric)
        
        best_model, best_rmse, flag = record_best_model_rmse(rsg, metric, best_model, best_rmse)
    
    best_model.fit(X_train_scaled.astype(np.float32), Y_train.astype(np.float32))
    return best_model, best_rmse

#the following 7 functions are different ways to get part of stacking data,
#nodes is the best hyperparameters for neural networks,
#X_train_scaled is the train data after feature scale,
#X_test_scaled is the test data after feature scale,
#n_folds is the fold number of the divided data,
#max_evals is the number of training,
#I recommend you use get_oof_validate1 or get_oof_validate2
def get_oof(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model(nodes, X_split_train, Y_split_train, max_evals)
            
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def get_oof_validate1(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model_validate1(nodes, X_split_train, Y_split_train, max_evals)
        
        #这里输出的是最佳模型的训练集和验证集上面的结果咯
        #很容易和上面的训练过程的最后一个输出重叠
        #这三个输出结果肯定是不一样的：
        #第一个输出和第二个输出的区别在于普通模型和最佳模型在训练集上面的输出
        #第二个输出和第三个输出的区别在于最佳模型在训练集和验证集上面的输出
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def get_oof_validate2(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model_validate2(nodes, X_split_train, Y_split_train, max_evals)
        
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)

        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def get_oof_noise_validate1(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model_noise_validate1(nodes, X_split_train, Y_split_train, max_evals)
        
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def get_oof_noise_validate2(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model_noise_validate2(nodes, X_split_train, Y_split_train, max_evals)
        
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def get_oof_noise_validate3(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model_noise_validate3(nodes, X_split_train, Y_split_train, max_evals)
        
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

def get_oof_noise_validate4(nodes, X_train_scaled, Y_train, X_test_scaled, n_folds = 5, max_evals = 10):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model, best_rmse= train_nn_model_noise_validate4(nodes, X_split_train, Y_split_train, max_evals)
        
        rmse1 = cal_nnrsg_rmse(best_model, X_split_train, Y_split_train)
        rmsle1 = cal_nnrsg_rmsle(best_model, X_split_train, Y_split_train)
        print(rmse1)
        print(rmsle1)
        train_rmse.append(rmse1)
        rmse2 = cal_nnrsg_rmse(best_model, X_split_valida, Y_split_valida)
        rmsle2 = cal_nnrsg_rmsle(best_model, X_split_valida, Y_split_valida)
        print(rmse2)
        print(rmsle2)
        valida_rmse.append(rmse2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

#the following 7 functions are different ways to get stacking data, neural network model stacking. 
#nodes_list is the list of best hyperparameters for neural networks,
#X_train_scaled is the train data after feature scale,
#X_test_scaled is the test data after feature scale,
#n_folds is the fold number of the divided data,
#max_evals is the number of training,
#I recommend using stacked_features_validate1 or stacked_features_validate2
#the first one can save training time, 40 and 25 are fine choice for the last two function parameters.
#the second one has less overfitting risk, 30 and 35 are fine choice for the last two function parameters.  
def stacked_features(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_validate1(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_validate1(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_validate2(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_validate2(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#我个人觉得这样的训练方式好像导致过拟合咯，所以采用下面的方式进行训练。
#每一轮进行get_oof_validate1的时候都增加了噪声，让每个模型都有所不同咯。
def stacked_features_noise_validate1(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
    
        #在这里增加一个添加噪声的功能咯,这里发现了一个BUG，居然数据没有被修改
        #不对，这个函数其实是对的，测试了半天发现了其实问题是出在参数上面咯
        X_noise_train, Y_noise_train = noise_augment_dataframe_data(nodes_list[0]["mean"], nodes_list[0]["std"], X_train_scaled, Y_train, columns=[])#columns=[i for i in range(1, 20)])
        #X_noise_train, Y_noise_train = noise_augment_dataframe_data(0, 0.02, X_train_scaled, Y_train, columns=[i for i in range(1, 20)])
        #X_train_scaled.to_csv("C:\\Users\\1\\Desktop\\temp2.csv", index=False)
        #X_noise_train.to_csv("C:\\Users\\1\\Desktop\\temp3.csv", index=False)
        #print(any(X_noise_train != X_train_scaled))
        
        oof_train, oof_test, best_model= get_oof_noise_validate1(nodes_list[i], X_noise_train.values, Y_noise_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#下面是我想到的第二种增加模型噪声的方式以防止过拟合咯。
#我个人觉得这样的训练方式好像导致过拟合咯，所以采用下面的方式进行训练。
#每一轮进行get_oof_validate1的时候都增加了噪声，让每个模型都有所不同咯。
def stacked_features_noise_validate2(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_noise_validate2(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_noise_validate3(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_noise_validate3(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

def stacked_features_noise_validate4(nodes_list, X_train_scaled, Y_train, X_test_scaled, folds, max_evals):
    
    input_train = [] 
    input_test = []
    nodes_num = len(nodes_list)
        
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_noise_validate4(nodes_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds, max_evals)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#useless function, I used it to try an idea but finally proved no use
def lr_stacking_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, max_evals=2000):
    
    rsg = LinearRegression()
    rsg.fit(stacked_train, Y_train)
    lr_pred = rsg.predict(stacked_test)

    save_best_model(rsg, nodes_list[0]["title"]+"_"+str(len(nodes_list)))
    #这个需要测试一下输出的是几位小数哦，别整奇怪的东西哦
    #Y_pred1 = (rsg.predict(stacked_test.values.astype(np.float32))).reshape(-1,1)
    #Y_pred2 = rsg.predict(stacked_test.values.astype(np.float32))
    #Y_pred3 = (rsg.predict(stacked_test.values.astype(np.float32))).flatten()
    #Y_pred4 = (rsg.predict(stacked_test.values.astype(np.float32))).reshape(1,-1)
    #只有下面的这种写法可以实现写入文件，我有点纠结pred是不是需要调用round函数呢？
    Y_pred = np.round((rsg.predict(stacked_test.values.astype(np.float32))).flatten(), decimals=1)

    data = {"Id":data_test["Id"], "SalePrice":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(nodes_list[0]["path"], index=False)
    print("prediction file has been written.")
    
    #这边的API确实是调用score。。
    #R^2 of self.predict(X) wrt. y.
    best_score= rsg.score(stacked_train, Y_train) 
    print(best_score)
    best_rmse = cal_nnrsg_rmse(rsg, stacked_train.values, Y_train.values)
    print(best_rmse)
    best_rmsle = cal_nnrsg_rmsle(rsg, stacked_train.values, Y_train.values)
    print(best_rmsle)
    return rsg, Y_pred

#lasso after randomized search and cross validation for stacking data prediction,
#nodes_list is the list of best hyperparameters for neural networks,
#data_test is the test data,
#stacked_train is the train data after stacking,
#stacked_test is the test data after stacking,
#max_evals is the iterations number of randomized search
def lasso_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, max_evals=8000):
    
    #Lasso回归的损失函数优化方法常用的有两种，坐标轴下降法和最小角回归法。
    #LassoLars类采用的是最小角回归法，前面讲到的Lasso类采用的是坐标轴下降法。
    #Lasso加上RandomizedSearchCV，基本上就大于LassoCV的，毕竟后者只是CV别个参数呢
    rsg = Lasso(max_iter=8000, tol=0.01)
    #这边的参数我觉得还是有必要进行这样的设置的，才能够比较好的表示
    param_dist = {"alpha": np.linspace(-3, 5, 100),
                  "fit_intercept": [True, False],
                  "normalize": [True, False],
                  "positive": [True, False],
                  }
    random_search = RandomizedSearchCV(rsg, param_distributions=param_dist, n_iter=max_evals)
    random_search.fit(stacked_train, Y_train)
    best_acc = random_search.best_estimator_.score(stacked_train, Y_train)
    lr_pred = random_search.best_estimator_.predict(stacked_test)

    save_best_model(random_search.best_estimator_, nodes_list[0]["title"]+"_"+str(len(nodes_list)))
    Y_pred = random_search.best_estimator_.predict(stacked_test.values.astype(np.float32))
            
    data = {"Id":data_test["Id"], "SalePrice":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(nodes_list[0]["path"], index=False)
    print("prediction file has been written.")
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    print()
    return random_search.best_estimator_, Y_pred

#useless function, I used it to try an idea but finally proved no use
def svm_stacking_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, max_evals=50):
    
    lsvr = svm.SVR(kernel='linear')
    param_dist = {"C": np.linspace(0.001, 100000, 10000)}
    random_search = RandomizedSearchCV(lsvr, param_distributions=param_dist, n_iter=max_evals)
    random_search.fit(stacked_train, Y_train)
    best_rmse= random_search.best_estimator_.score(stacked_train, Y_train)
    lr_pred = random_search.best_estimator_.predict(stacked_test)

    save_best_model(random_search.best_estimator_, nodes_list[0]["title"]+"_"+str(len(nodes_list)))
    Y_pred = random_search.best_estimator_.predict(stacked_test.values.astype(np.float32))
            
    data = {"Id":data_test["Id"], "SalePrice":Y_pred}
    output = pd.DataFrame(data = data)
            
    output.to_csv(nodes_list[0]["path"], index=False)
    print("prediction file has been written.")
     
    print("the best accuracy rate of the model on the whole train dataset is:", best_rmse)
    print()
    return random_search.best_estimator_, Y_pred
    
#现在直接利用经验参数值进行搜索咯，这样可以节约计算资源
#space is the optimization space of bayesian hyperparameters
space = {"title":hp.choice("title", ["stacked_house_prices"]),
         "path":hp.choice("path", ["House_Prices_Prediction.csv"]),
         "mean":hp.choice("mean", [0]),
         "std":hp.choice("std", [0]),
         "max_epochs":hp.choice("max_epochs",[3000]),
         "patience":hp.choice("patience", [3,6,9]),
         "lr":hp.choice("lr", [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010,
                               0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020,
                               0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
                               0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040,
                               0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050,
                               0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060,
                               0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070,
                               0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080,
                               0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
                               0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100,
                               0.00101, 0.00102, 0.00103, 0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.00110,
                               0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.00120,
                               0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00129, 0.00130,
                               0.00131, 0.00132, 0.00133, 0.00134, 0.00135, 0.00136, 0.00137, 0.00138, 0.00139, 0.00140,
                               0.00141, 0.00142, 0.00143, 0.00144, 0.00145, 0.00146, 0.00147, 0.00148, 0.00149, 0.00150,
                               0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00158, 0.00159, 0.00160,
                               0.00161, 0.00162, 0.00163, 0.00164, 0.00165, 0.00166, 0.00167, 0.00168, 0.00169, 0.00170,
                               0.00171, 0.00172, 0.00173, 0.00174, 0.00175, 0.00176, 0.00177, 0.00178, 0.00179, 0.00180]),  
         "optimizer__weight_decay":hp.choice("optimizer__weight_decay",[0.000, 0.00000001, 0.000001, 0.0001, 0.01]),  
         "criterion":hp.choice("criterion", [RMSELoss]),

         "batch_size":hp.choice("batch_size", [128]),
         "optimizer__betas":hp.choice("optimizer__betas",
                                      [[0.88, 0.9991], [0.88, 0.9993], [0.88, 0.9995], [0.88, 0.9997], [0.88, 0.9999],
                                       [0.90, 0.9991], [0.90, 0.9993], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999],
                                       [0.92, 0.9991], [0.92, 0.9993], [0.92, 0.9995], [0.92, 0.9997], [0.92, 0.9999]]),
         "input_nodes":hp.choice("input_nodes", [292]),
         "hidden_layers":hp.choice("hidden_layers", [1, 3, 5, 7, 9, 11]), 
         "hidden_nodes":hp.choice("hidden_nodes", [300, 350, 400, 450, 500, 550, 600, 650, 700, 
                                                   750, 800, 850, 900, 950, 1000, 1050, 1100]), 
         "output_nodes":hp.choice("output_nodes", [1]),
         "percentage":hp.choice("percentage", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]),
         "weight_mode":hp.choice("weight_mode", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
         #这个按照deep learning那本书的说法，应该是设置为靠近0的很小的常数值
         #但是这个区间我有点拿不准，所以我可能要参考一下之前赋值方式的取值范围咯
         #我按照赋值的方式对于偏置的范围进行计算，偏置几乎不会大于[-0.1, 0.1]
         #我测算了一下，按照普通的计算方式，下面的取值范围应该是比较合适的吧
         "bias":hp.choice("bias", [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]),
         "device":hp.choice("device", ["cpu"]),
         "optimizer":hp.choice("optimizer", [torch.optim.Adam])
         }

#space_nodes is the optimization space of bayesian hyperparameters, 
#which mainly used to and makes it easier to get the best hyperparameters
space_nodes = {"title":["stacked_house_prices"],
               "path":["House_Prices_Prediction.csv"],
               "mean":[0],
               "std":[0],
               "max_epochs":[3000],
               "patience":[3,6,9],
               "lr":[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010,
                     0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020,
                     0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
                     0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040,
                     0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050,
                     0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060,
                     0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070,
                     0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080,
                     0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
                     0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100,
                     0.00101, 0.00102, 0.00103, 0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.00110,
                     0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.00120,
                     0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00129, 0.00130,
                     0.00131, 0.00132, 0.00133, 0.00134, 0.00135, 0.00136, 0.00137, 0.00138, 0.00139, 0.00140,
                     0.00141, 0.00142, 0.00143, 0.00144, 0.00145, 0.00146, 0.00147, 0.00148, 0.00149, 0.00150,
                     0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00158, 0.00159, 0.00160,
                     0.00161, 0.00162, 0.00163, 0.00164, 0.00165, 0.00166, 0.00167, 0.00168, 0.00169, 0.00170,
                     0.00171, 0.00172, 0.00173, 0.00174, 0.00175, 0.00176, 0.00177, 0.00178, 0.00179, 0.00180],
               "optimizer__weight_decay":[0.000, 0.00000001, 0.000001, 0.0001, 0.01],
               "criterion":[RMSELoss],
               "batch_size":[128],
               "optimizer__betas":[[0.88, 0.9991], [0.88, 0.9993], [0.88, 0.9995], [0.88, 0.9997], [0.88, 0.9999],
                                   [0.90, 0.9991], [0.90, 0.9993], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999],
                                   [0.92, 0.9991], [0.92, 0.9993], [0.92, 0.9995], [0.92, 0.9997], [0.92, 0.9999]],
               "input_nodes":[292],
               "hidden_layers":[1, 3, 5, 7, 9, 11], 
               "hidden_nodes":[300, 350, 400, 450, 500, 550, 600, 650, 700, 
                               750, 800, 850, 900, 950, 1000, 1050, 1100], 
               "output_nodes":[1],
               "percentage":[0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
               "weight_mode":[1, 2, 3, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13],
               "bias":[-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03],
               "device":["cpu"],
               "optimizer":[torch.optim.Adam]
               }

#其实本身不需要best_nodes主要是为了快速测试
#不然每次超参搜索的best_nodes效率太低了吧
#best_nodes used to record the best hyperparameters of the neural networks
best_nodes = {"title":"stacked_house_prices",
              "path":"House_Prices_Prediction.csv",
              "mean":0,
              "std":0,
              "max_epochs":3000,
              "patience":5,
              "lr":0.00010,
              "optimizer__weight_decay":0.005,
              "criterion":RMSELoss,
              "batch_size":128,
              "optimizer__betas":[0.86, 0.999],
              "input_nodes":292,
              "hidden_layers":3,
              "hidden_nodes":300,
              "output_nodes":1,
              "percentage":0.2,
              "weight_mode":1,
              "bias":0.0,
              "device":"cpu",
              "optimizer":torch.optim.Adam
              }

#run the following code for running environment test
#reshape data for RMSELoss, or you will get error
#split train data and validation data for best hyperparameters selection.
Y_train_temp = Y_train.values.reshape(-1,1)
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=['SalePrice'])
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.14)

start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
#max_evals determine hyperparameters search times, bigger max_evals may lead to better results.
best_params = fmin(nn_f, space, algo=algo, max_evals=120, trials=trials)

#save the result of the hyperopt(bayesian optimization) search.
best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "house_price")

#use 2 best nodes to create 2 neural network model for stacking.
nodes_list = [best_nodes, best_nodes]
#the following code can change the settings of the stacking process
#you may use it as following when you need.
#change settings except device or path is not recommended, cause you may lost best hyperparameters.
#if you must use specific parameters, set it and start hyperparameters once more.
#for item in nodes_list:
#    item["device"] = "cpu" #set the device to train neural network, "cpu" means using cpu, "cuda" means using gpu.
#    item["batch_size"] = 256 #set the batch_size of the neural network.
#    item["path"] = "House_Prices_Prediction.csv" #set the file path of the prediction file.
#neural network model stacking. I recommend using stacked_features_validate1 or stacked_features_validate2
#the first one can save training time, 40 and 25 are fine choice for the last two function parameters.
#the second one has less overfitting risk, 30 and 35 are fine choice for the last two function parameters.  
stacked_train, stacked_test = stacked_features_validate1(nodes_list, X_train_scaled, Y_train, X_test_scaled, 2, 2)
#save the stacking intermediate result.
save_stacked_dataset(stacked_train, stacked_test, "house_price")

#reshape data for RandomizedSearchCV, or you will get error
Y_train_temp = Y_train.values.flatten()
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=['SalePrice'])
#use lasso to fit stacked_train/stacked_test and predict the result. 
lasso_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 600)
#print time cost
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

"""
#run the following code for neural network model training and prediction.
#use hyperopt(bayesian optimization) to search the best network structure.
#have a look at hyperopt will help in understanding the following code.
#split train data and validation data for best hyperparameters selection.
Y_train_temp = Y_train.values.reshape(-1,1)
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=['SalePrice'])
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.14)

start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
#max_evals determine hyperparameters search times, bigger max_evals may lead to better results.
best_params = fmin(nn_f, space, algo=algo, max_evals=2200, trials=trials)

#save the result of the hyperopt(bayesian optimization) search.
best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "house_price")

#use 5 best nodes to create 5 neural network model for stacking.
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
#the following code can change the settings of the stacking process
#you may use it as following when you need.
#change settings except device or path is not recommended, cause you may lost best hyperparameters.
#if you must use specific parameters, set it and start hyperparameters once more.
#for item in nodes_list:
#    item["device"] = "cpu" #set the device to train neural network, "cpu" means using cpu, "cuda" means using gpu.
#    item["batch_size"] = 256 #set the batch_size of the neural network.
#    item["path"] = "House_Prices_Prediction.csv" #set the file path of the prediction file.
#neural network model stacking. I recommend using stacked_features_validate1 or stacked_features_validate2
#the first one can save training time, 40 and 25 are fine choice for the last two function parameters.
#the second one has less overfitting risk, 30 and 35 are fine choice for the last two function parameters.  
stacked_train, stacked_test = stacked_features_validate1(nodes_list, X_train_scaled, Y_train, X_test_scaled, 40, 25)
#save the stacking intermediate result.
save_stacked_dataset(stacked_train, stacked_test, "house_price")

#reshape data for RandomizedSearchCV, or you will get error
Y_train_temp = Y_train.values.flatten()
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=['SalePrice'])
#use lasso to fit stacked_train/stacked_test and predict the result. 
lasso_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 600)
#print time cost
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""

"""
#run the following code for neural network model training and prediction.
#use hyperopt(bayesian optimization) to search the best network structure.
#have a look at hyperopt will help in understanding the following code.
#split train data and validation data for best hyperparameters selection.
Y_train_temp = Y_train.values.reshape(-1,1)
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=['SalePrice'])
X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train_scaled, Y_train, test_size=0.14)

start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
#max_evals determine hyperparameters search times, bigger max_evals may lead to better results.
best_params = fmin(nn_f, space, algo=algo, max_evals=2200, trials=trials)

#save the result of the hyperopt(bayesian optimization) search.
best_nodes = parse_nodes(trials, space_nodes)
save_inter_params(trials, space_nodes, best_nodes, "house_price")

#use 5 best nodes to create 5 neural network model for stacking.
nodes_list = [best_nodes, best_nodes, best_nodes, best_nodes, best_nodes]
#the following code can change the settings of the stacking process
#you may use it as following when you need.
#change settings except device or path is not recommended, cause you may lost best hyperparameters.
#if you must use specific parameters, set it and start hyperparameters once more.
#for item in nodes_list:
#    item["device"] = "cpu" #set the device to train neural network, "cpu" means using cpu, "cuda" means using gpu.
#    item["batch_size"] = 256 #set the batch_size of the neural network.
#    item["path"] = "House_Prices_Prediction.csv" #set the file path of the prediction file.
#neural network model stacking. I recommend using stacked_features_validate1 or stacked_features_validate2
#the first one can save training time, 40 and 25 are fine choice for the last two function parameters.
#the second one has less overfitting risk, 30 and 35 are fine choice for the last two function parameters.  
stacked_train, stacked_test = stacked_features_validate2(nodes_list, X_train_scaled, Y_train, X_test_scaled, 30, 35)
#save the stacking intermediate result.
save_stacked_dataset(stacked_train, stacked_test, "house_price")

#reshape data for RandomizedSearchCV, or you will get error
Y_train_temp = Y_train.values.flatten()
Y_train = pd.DataFrame(data=Y_train_temp.astype(np.float32), columns=['SalePrice'])
#use lasso to fit stacked_train/stacked_test and predict the result. 
lasso_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 600)
#print time cost
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
"""
