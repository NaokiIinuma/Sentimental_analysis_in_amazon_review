from sklearn.metrics import classification_report
import pickle
from model import model
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

def foward(model,foward_name,input):
    if foward_name == "CNN_vanilla":
        return model.CNN_vanilla(input)
    elif foward_name == "CNN_multi_filter":
        return model.CNN_multi_filter(input)

def eval(model,val_idx,input_id,y_true,device,foward_name):
    model.eval()
    e_batch=1024
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
        output=foward(model,foward_name,batch_val_idx_tensor)
        for o in output:
            val_pred_y_all.append(int(torch.argmax(o)))
    report_dict=classification_report(val_y_all, val_pred_y_all, output_dict=True, target_names=["☆","☆ ☆","☆ ☆ ☆","☆ ☆ ☆ ☆","☆ ☆ ☆ ☆ ☆"])
    print(classification_report(val_y_all, val_pred_y_all, digits=3, target_names=["☆","☆ ☆","☆ ☆ ☆","☆ ☆ ☆ ☆","☆ ☆ ☆ ☆ ☆"]))

def train(model,batch,EPOCH,device,data,dataset_idx,optimizer,foward_name):
    input_id,y_true = data
    train_idx,val_idx,test_idx = dataset_idx
    model.train()
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
            output=foward(model,foward_name,batch_input_idx_tensor)
            loss=criterion(output,train_y_tensor)
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch {} : {}".format(n_epoch,np.array(loss_epoch).mean()))
        if n_epoch%10==0:
            eval(model,val_idx,input_id,y_true,device,foward_name)
            model.train()

if  __name__ in "__main__":
    with open("input_id.pkl","rb") as f:
        input_id=pickle.load(f)
    with open("y_true.pkl","rb") as f:
        y_true=pickle.load(f)
    with open("word2id.pkl","rb") as f:
        word2id=pickle.load(f)
    with open("train_idx.pkl","rb") as f:
        train_idx=pickle.load(f)
    with open("val_idx.pkl","rb") as f:
        val_idx=pickle.load(f)
    with open("test_idx.pkl","rb") as f:
        test_idx=pickle.load(f)
    batch=512
    EPOCH=101
    model=model(len(word2id)+1)
    device="cuda:0"
    foward_dict=["CNN_vanilla","CNN_multi_filter"]
    foward_name=foward_dict[1]
    model.to(device)
    optimizer=optim.Adam(model.parameters())
    train(model,batch,EPOCH,device,[input_id,y_true],[train_idx,val_idx,test_idx],optimizer,foward_name)