#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from torchviz import make_dot
from sklearn import preprocessing


# In[2]:

training_data_path = './Data1/training.csv'
validation_data_path = './Data1/validation.csv'
test_data_path = './Data1/test.csv'


Raw_data = pd.read_csv(training_data_path, sep=',')#load the training dataset

print(Raw_data.shape)
print(Raw_data.head())
print('')
# print(index.shape)
# print(index.tail(20))


# In[3]:


# 均值填充
Clean_data = Raw_data
Clean_data['a10'] = Raw_data['a10'].fillna(Raw_data['a10'].mean())
Clean_data['a11'] = Raw_data['a11'].fillna(Raw_data['a11'].mean())
Clean_data['a13'] = Raw_data['a13'].fillna(Raw_data['a13'].mean())
Clean_data['a15'] = Raw_data['a15'].fillna(Raw_data['a15'].mean())
Clean_data['a16'] = Raw_data['a16'].fillna(Raw_data['a16'].mean())
Clean_data['a17'] = Raw_data['a17'].fillna(Raw_data['a17'].mean())

Clean_data.isnull().sum() 
print(type(Clean_data))
# Raw_data.head()#print the first 5 colunms


# In[4]:


X_train = Clean_data.iloc[:, 0:17].values
y_train = Clean_data['class label'].values

print(X_train.shape)
print(y_train.shape)


# In[5]:


# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
print((X_train.max(axis=0) > (1-1e-7)).sum(), (X_train.min(axis=0) == 0).sum())
print(type(X_train))


# In[6]:


colname=Clean_data.columns.values[:-1]
plt.figure(figsize=(20,14))
for ii in range(len(colname)):
    plt.subplot(7,9,ii+1)
    sns.boxplot(x=y_train,y=X_train[:,ii])
    plt.title(colname[ii])
plt.subplots_adjust(hspace=0.4)
plt.show()


# In[7]:


# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.out = torch.nn.Linear(n_hidden, n_output)

#     def forward(self, x):
#         x = torch.nn.functional.relu(self.hidden(x))
#         x = self.out(x)
#         return x

# net = Net(17, 10, 8)
# print(net)


# In[8]:


# optimizer = torch.optim.Adam(net.parameters(), lr=0.5)
# loss_func = torch.nn.MSELoss()  # 均方差


# In[9]:


# X_train=torch.from_numpy(X_train.astype(np.float32))
# y_train=torch.from_numpy(y_train.astype(np.float32))



# In[10]:


# X_train = torch.unsqueeze(torch.FloatTensor(X_train), dim=1)
# y_train = torch.unsqueeze(torch.FloatTensor(y_train), dim=1)

# print(X_train.shape)
# print(y_train.shape)


# In[11]:


# for t in range(10000):
#     prediction = net(X_train)

#     loss = loss_func(prediction, y_train)  # 一定要prediction在前, y在后

#     optimizer.zero_grad()  # 梯度降零
#     loss.backward()
#     optimizer.step()
#     if t % 25 == 0:
# 	    # plot and show learning process
#         print('Loss=%.4f' % loss.data.numpy())



# In[12]:


class FullyConnectedNuralNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNuralNetwork,self).__init__()
        self.hidden1=nn.Sequential(
                nn.Linear(in_features=17,out_features=15,bias=True),
                nn.ReLU())
        self.hidden2=nn.Sequential(
                nn.Linear(in_features=15,out_features=8,bias=True),
                nn.ReLU())
#         self.hidden3=nn.Sequential(
#                 nn.Linear(in_features=13,out_features=8,bias=True),
#                 nn.ReLU())
#                 nn.Sigmoid())
            
    def forward(self, x):
        fc1 = self.hidden1(x)
#         fc2=self.hidden2(fc1)
        output = self.hidden2(fc1)
        return fc1, output


FCNN1 = FullyConnectedNuralNetwork()
# x=torch.randn(size=(1,17)).requires_grad_(True)
# y=FCNN1(x)
# FCArchitecture=make_dot(y,params=dict(list(FCNN1.named_parameters())+[('x',x)]))
# FCArchitecture.format='png'
# FCArchitecture.directory='../pic01/'
# FCArchitecture.view()


# In[13]:


X_train=torch.from_numpy(X_train.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))

train_data=Data.TensorDataset(X_train,y_train)
train_loader=Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=1)
for step,(batch_x,batch_y) in enumerate(train_loader):
    if step>0:
        break
print(step,batch_x.shape,batch_y.shape)



# In[14]:


# optimal
optomizerAdam=torch.optim.Adam(FCNN1.parameters(),lr=0.005)
lossFunc=nn.CrossEntropyLoss()


# In[15]:


history1=hl.History()
canvas1=hl.Canvas()
logStep=25
for epoch in range(60):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        _,output=FCNN1(batch_x)
#         print(batch_x.size())
#         print(batch_y.size())
        train_loss=lossFunc(output,batch_y.long())
        optomizerAdam.zero_grad()
        train_loss.backward()
        optomizerAdam.step()
        
        niter=epoch*len(train_loader)+step+1
        if niter % logStep ==0:
#             _,_,output=FCNN1(X_test)
            _,pre_lab=torch.max(output,1)
#             test_accuracy=accuracy_score(y_test,pre_lab)
            history1.log(niter,train_loss=train_loss)
            with canvas1:
                canvas1.draw_plot(history1['train_loss'])
#                 canvas1.draw_plot(history1['test_accuracy'])


# In[62]:


# Validation 
from sklearn.metrics import f1_score

valid_data=pd.read_csv('BigData Data1/validation.csv',sep=',')

valid_data.isnull().sum() 

valid_data['a10'] = valid_data['a10'].fillna(valid_data['a10'].mean())
valid_data['a11'] = valid_data['a11'].fillna(valid_data['a11'].mean())
valid_data['a13'] = valid_data['a13'].fillna(valid_data['a13'].mean())
valid_data['a15'] = valid_data['a15'].fillna(valid_data['a15'].mean())
valid_data['a16'] = valid_data['a16'].fillna(valid_data['a16'].mean())
valid_data['a17'] = valid_data['a17'].fillna(valid_data['a17'].mean())
print(valid_data.isnull().sum())


# In[63]:


X_valid = valid_data.iloc[:, 0:17].values
y_valid = valid_data['class label'].values

scaler=MinMaxScaler(feature_range=(0,1))
X_valid=scaler.fit_transform(X_valid)


X_valid=torch.from_numpy(X_valid.astype(np.float32))
y_valid=torch.from_numpy(y_valid.astype(np.float32))

_,y_predict = FCNN1(X_valid)


# In[64]:


print(type(y_predict))
print(y_predict.shape)
print(type(y_valid))
print(y_valid.shape)


# In[65]:


# y_predict = y_predict.detach().numpy()
# y_predict = np.around(y_predict,0).astype(int)
y_predict = y_predict.array()


# In[58]:


y_valid = y_valid.numpy()
y_valid = np.around(y_valid,0).astype(int)
print(y_valid)
print(y_predict)


# In[59]:


print(y_predict[2])
print(y_valid.shape)
print(y_predict.shape)


# In[60]:


from sklearn.metrics import classification_report

print(classification_report(y_valid, y_predict))


# In[52]:


print(f1_score(y_valid, y_predict, average='macro'))


# In[ ]:




