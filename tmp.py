# %%
import numpy as np 
import pandas as pd
import os,sys

path1 = "./data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
path2 = "./data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
data1=pd.read_csv(path1)
data2=pd.read_csv(path2)
# %%
data_tmp1 = data1[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]
data_tmp2 = data2[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]
data_tmp = data_tmp2
data_used= data_tmp[data_tmp["reviews.rating"].notnull()]
n_rate=data_used["reviews.rating"].value_counts()
print(n_rate)
# %%
y=data_used["reviews.rating"]
x=data_used[['reviews.text' , 'reviews.title' , 'reviews.username']]

import sklearn.model_selection as MS
from sklearn.model_selection import KFold

def calc_maen_var(y,idx):
    tmp_y=y[idx]
    print('__mean__')
    print(np.mean(tmp_y))
    print('__variance__')
    print(np.var(tmp_y))

kfold = KFold(5, shuffle=True)
cvs = list(kfold.split(y, y))
cv_val_idxs = [list(val_idx) for train_idx, val_idx in cvs]
train_idx = sum(cv_val_idxs[:3], [])
val_idx = cv_val_idxs[3]
test_idx = cv_val_idxs[4]

print("train_data")
calc_maen_var(y,train_idx)
print("val_data")
calc_maen_var(y,val_idx)
print("test_data")
calc_maen_var(y,test_idx)
# %%
import nltk
nltk.download('punkt')
class preprocess():
    def __init__(self, data_used):
        self.data = data_used
        self.wordpool = set()
        self.word2id = dict()
        self.id2word = dict()
        self.max_x_len = -1
    def make_wordpool(self):
        for x in zip(data_used['reviews.title'],data_used['reviews.text']):
            x_title=str(x[0])
            x_sentence=str(x[1])
            x_title_sp=nltk.word_tokenize(x_title)
            x_sentence_sp=nltk.word_tokenize(x_sentence)
            x_len = len(x_title_sp)+len(x_sentence_sp)
            if self.max_x_len<x_len:
                self.max_x_len=x_len
            self.wordpool = self.wordpool | set(x_title_sp)
            self.wordpool = self.wordpool | set(x_sentence_sp)
    def make_word2id_and_id2word(self):
        for i,w in enumerate(list(self.wordpool)):
            self.word2id[w]=i+1
            self.id2word[i+1]=w
    def tokenize(self,data_used,pading):
        input = list()
        for x in zip(data_used['reviews.title'],data_used['reviews.text']):
            x_title=str(x[0])
            x_sentence=str(x[1])
            x_title_sp=nltk.word_tokenize(x_title)
            x_sentence_sp=nltk.word_tokenize(x_sentence)
            x_title_sp.extend(x_sentence_sp)
            tmp = list()
            for w in x_title_sp:
                tmp.append(self.word2id[w])
            if pading:
                for i in range(self.max_x_len-len(tmp)):
                    tmp.append(0)
            if tmp==[]:
                sys.exit()
            input.append(tmp)
        return input
    def get_word2id(self):
        return self.word2id
    def get_id2word(self):
        return self.id2word
    def fix_y(self,y):
        tmp=list()
        for t in y:
            tmp.append(t-1)
        return tmp
prep=preprocess(data_used)
prep.make_wordpool()
prep.make_word2id_and_id2word()
input_id=prep.tokenize(data_used,pading=True)
y_true=prep.fix_y(y)
# %%
word2id=prep.get_word2id()
print(len(word2id))
# %%
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class CNN(nn.Module):
    
    def __init__(self,max_features):
        super(CNN, self).__init__()
        filter_sizes1 = 3
        filter_sizes2 = 5
        num_filters = 50
        self.embedding = nn.Embedding(max_features, 50, padding_idx=0)
        self.conv1_1 = nn.Conv1d(50, num_filters, filter_sizes1)
        self.conv1_2 = nn.Conv1d(50, num_filters, filter_sizes2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1*num_filters*2, 5)
    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x))
        x1 = F.max_pool1d(x1,x1.size(2))
        x2 = F.max_pool1d(x2,x2.size(2))
        x1 = torch.squeeze(x1,2)
        x2 = torch.squeeze(x2,2)
        x = torch.cat([x1,x2],1)
        x = self.dropout(x)  
        output = self.fc1(x)  
        return output

from sklearn.metrics import classification_report
def eval(model,val_idx,input_id,y_true,device):
    model.eval()
    e_batch=512
    val_y_all=list()
    val_pred_y_all=list()
    for n_kousin in range(len(val_idx)//e_batch+1):
        batch_input_idx=val_idx[n_kousin*e_batch:min((n_kousin+1)*e_batch,len(train_idx))]
        val_input=list()
        val_y=list()
        for idx in batch_input_idx:
            val_input.append(input_id[idx])
            val_y.append(y_true[idx])
        val_y_all.extend(val_y)
        batch_val_idx_tensor=torch.tensor(val_input).to(device)
        val_y_tensor=torch.tensor(val_y).to(device)
        output=model(batch_val_idx_tensor)
        for o in output:
            val_pred_y_all.append(int(torch.argmax(o)))
    print(classification_report(val_y_all, val_pred_y_all,target_names=["☆","☆ ☆","☆ ☆ ☆","☆ ☆ ☆ ☆","☆ ☆ ☆ ☆ ☆"]))
batch=64
EPOCH=51
model=CNN(len(word2id)+1)
device="cuda:0"
model.to(device)
model.train()
optimizer=optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
for n_epoch in range(EPOCH):
    random.shuffle(train_idx)
    n=0
    optimizer.zero_grad()
    loss_epoch=list()
    for n_kousin in range(len(train_idx)//batch+1):
        batch_input_idx=train_idx[n_kousin*batch:min((n_kousin+1)*batch,len(train_idx))]
        train_input=list()
        train_y=list()
        for idx in batch_input_idx:
            train_input.append(input_id[idx])
            train_y.append(y_true[idx])
        batch_input_idx_tensor=torch.tensor(train_input).to(device)
        train_y_tensor=torch.tensor(train_y).to(device)
        output=model(batch_input_idx_tensor)
        loss=criterion(output,train_y_tensor)
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("epoch {} : {}".format(n_epoch,np.array(loss_epoch).mean()))
    if n_epoch%10==0:
        eval(model,val_idx,input_id,y_true,device)
        model.train()