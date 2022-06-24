#coding: utf8
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn import manifold
import numpy as np

from process import *
import process

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    
def show_loss(epochs, train_loss_list, test_loss_list):
    x1 = range(0,epochs)
    x2 = range(0,epochs)
    plt.figure()
    plt.plot(x1, train_loss_list, c='blue', label="Train_Loss")
    plt.plot(x2, test_loss_list, c='red', label="Test_Loss")
    plt.legend(loc='best')
    plt.show()
    
def show_acc(epochs, train_acc_list, test_acc_list):
    x1 = range(0,epochs)
    x2 = range(0,epochs)
    plt.figure()
    plt.plot(x1, train_acc_list, c='blue',label="Train_Acc")
    plt.plot(x2, test_acc_list, c='red',label="Test_Acc")
    plt.legend(loc='best')
    plt.show()
    
def save_model(accs, f1, model, modelname, save_path, fold=None):
    if fold:
        torch.save(model.state_dict(),os.path.join(save_path, modelname+'-'+str(accs)+'-'+str(F1)+'.th'))
    else:  
        torch.save(model.state_dict(),os.path.join(save_path, fold + modelname+'-'+str(accs)+'-'+str(F1)+'.th'))
        
        
        
        
        
        
        
        
        
        