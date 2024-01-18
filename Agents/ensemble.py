from lasso_inf import lasso
from TransPath.CT_inference import CTrans
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import ast
from HGCN.HGCN_code.test import HGCN

print("HGCN")
a = HGCN()
print("a",a)

print("Lasso")
X,Y = lasso()
print(X,Y)

test_data = ['TCGA-49-4501','TCGA-49-6742','TCGA-75-7030','TCGA-86-6562']
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])
X2,Y2 = CTrans(test_data)
print("TransPath")
print(X2,Y2)