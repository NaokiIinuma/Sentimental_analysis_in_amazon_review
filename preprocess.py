import numpy as np 
import pandas as pd
import sklearn.model_selection as MS
from sklearn.model_selection import KFold
import nltk
import pickle
nltk.download('punkt')

def calc_maen_var(y,idx):
    tmp_y=y[idx]
    print('__mean__')
    print(np.mean(tmp_y))
    print('__variance__')
    print(np.var(tmp_y))

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

path1 = "./data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
path2 = "./data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
data1=pd.read_csv(path1)
data2=pd.read_csv(path2)

data_tmp1 = data1[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]
data_tmp2 = data2[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]
data_tmp = data_tmp2
data_used= data_tmp[data_tmp["reviews.rating"].notnull()]
n_rate=data_used["reviews.rating"].value_counts()
print(n_rate)

y=data_used["reviews.rating"]
x=data_used[['reviews.text' , 'reviews.title' , 'reviews.username']]

kfold = KFold(5, shuffle=True)
cvs = list(kfold.split(y, y))
cv_val_idxs = [list(val_idx) for train_idx, val_idx in cvs]
train_idx = sum(cv_val_idxs[:3], [])
val_idx = cv_val_idxs[3]
test_idx = cv_val_idxs[4]

print("\ntrain_data")
calc_maen_var(y,train_idx)
print("\nval_data")
calc_maen_var(y,val_idx)
print("\ntest_data")
calc_maen_var(y,test_idx)

prep=preprocess(data_used)
prep.make_wordpool()
prep.make_word2id_and_id2word()
input_id=prep.tokenize(data_used,pading=True)
y_true=prep.fix_y(y)
word2id=prep.get_word2id()
with open("input_id.pkl","wb") as f:
    pickle.dump(input_id,f)
with open("y_true.pkl","wb") as f:
    pickle.dump(y_true,f)
with open("word2id.pkl","wb") as f:
    pickle.dump(word2id,f)
with open("train_idx.pkl","wb") as f:
    pickle.dump(train_idx,f)
with open("val_idx.pkl","wb") as f:
    pickle.dump(val_idx,f)
with open("test_idx.pkl","wb") as f:
    pickle.dump(test_idx,f)