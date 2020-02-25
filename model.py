import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class model(nn.Module):
    
    def __init__(self,max_features):
        super(model, self).__init__()
        filter_sizes1 = 3
        filter_sizes2 = 5
        num_filters = 100
        dropout_prob = 0.3
        self.embedding = nn.Embedding(max_features, 50, padding_idx=0)
        self.conv1_1 = nn.Conv1d(50, num_filters, filter_sizes1)
        self.conv1_2 = nn.Conv1d(50, num_filters, filter_sizes2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1_1 = nn.Linear(1*num_filters*2, 5)
        self.fc1_2 = nn.Linear(1*num_filters, 5)
    def CNN_multi_filter(self, x):
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
        output = self.fc1_1(x)  
        return output
    
    def CNN_vanilla(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.conv1_1(x))
        x = F.max_pool1d(x,x.size(2))
        x = torch.squeeze(x,2)
        x = self.dropout(x)  
        output = self.fc1_2(x)
        return output