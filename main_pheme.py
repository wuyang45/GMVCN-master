#coding: utf8
import sys,os
from utils import *
from config_pheme import args

import torch
import torch.nn as nn 
from torch_scatter import scatter_mean, scatter_max
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import max_pool
from tqdm import tqdm
import torch.optim as optim
import time
from tabulate import tabulate
from sklearn.metrics import classification_report
import random
from torch.autograd import Variable

from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv
import copy
import matplotlib.pyplot as plt
import math
import json

class TDGCN(torch.nn.Module):
    def __init__(self, args):
        super(TDGCN, self).__init__()
        self.conv1 = GCNConv(args.in_dim, args.hid_dim)
        self.conv2 = GCNConv(args.hid_dim, args.out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x
        
class BUGCN(torch.nn.Module):
    def __init__(self, args):
        super(BUGCN, self).__init__()
        self.conv1 = GCNConv(args.in_dim, args.hid_dim)
        self.conv2 = GCNConv(args.hid_dim, args.out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x
        
class Co_CNN(nn.Module):
    def __init__(self, args, channel=2):
        super(Co_CNN, self).__init__()
        
        self.convs = nn.ModuleList([nn.Conv2d(channel, args.Knum, (int(K), args.out_dim)) for K in args.Ks])
        
    def forward(self, x, batch):
        batch_size = max(batch) + 1
        hub = []
        mask = torch.tensor([True]).to(args.device)
        first = 0
        second = 0
        x = x.permute(1,0,2) #[nodes*batch_s, 2, out_dim]
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            #print('index:{}'.format(index))
            first = second
            count = 0
            for j in index:
                if j == mask:
                    count += 1
            second = first + count
            #batch_index = torch.tensor([first, second])
            #x_batch = torch.index_select(x, 0, batch_index) #[this_batch, 2, in_dim]
            x_batch = x[first:second]
            #print('x_batch:{}'.format(x_batch.shape))
            x_batch = x_batch.permute(1,0,2)
            x_batch = x_batch.unsqueeze(0) #[1, 2, this_batch, in_dim]
            #print('x_batch:{}'.format(x_batch.shape))
            x_batch = [F.relu(conv(x_batch)).squeeze(3) for conv in self.convs] # len(Ks)*(1, Knum, this_batch)
            #print('x_batch_cnn:{}'.format(x_batch[0].shape))
            x_batch = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x_batch] #len(Ks)*(1, Knum)
            x_batch = torch.cat(x_batch, 1) #[1, Knum*len(Ks)]
            hub.append(x_batch)
            
        x_new = torch.cat(hub, 0)
        return x_new #[batch_size, Knum*len(Ks)]
        
class GMVCN(torch.nn.Module):
    def __init__(self, args):
        super(GMVCN, self).__init__()
        self.TDrumorGCN = TDGCN(args)
        self.BUrumorGCN = BUGCN(args)
        self.co_CNN = Co_CNN(args)
        self.fc = torch.nn.Linear(args.Knum*len(args.Ks), 3) 
    
    def forward(self, data):
        TD_x = self.TDrumorGCN(data).unsqueeze(0)
        BU_x = self.BUrumorGCN(data).unsqueeze(0)
        x = torch.cat((BU_x,TD_x), 0)
        x = self.co_CNN(x, data.batch)
            
        x=self.fc(x)
        pred = F.softmax(x, dim=1)
        return x, pred
    
def train(fold, args, datapath, treeLenDic, save_path, x_train, x_test,):
    
    model = GMVCN(args).to(args.device)
    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = torch.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':args.lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr/5}
        ], lr=args.lr, weight_decay=args.w_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    train_data_list, test_data_list = process.loadDataPheme(datapath, treeLenDic, x_train, x_test, args.dropedge, args.lower)
    
    for epoch in range(args.max_epochs):
        train_loader = DataLoader(train_data_list, batch_size = args.batchsize, shuffle = True)
        test_loader = DataLoader(test_data_list, batch_size = args.batchsize)
        
        model.train()
        t = time.time()
        train_epoch_loss = 0
        train_epoch_acc = 0
        for i, train_data in enumerate(train_loader):
            train_data.to(args.device)
            train_label = train_data['y'].type(torch.LongTensor).to(args.device)
        
            optimizer.zero_grad()
            output, pred = model(train_data)
            loss_train = criterion(output, train_label)
            loss_train.backward(retain_graph=True)
            optimizer.step()
        
            acc_train = accuracy(pred, train_label) 
            train_epoch_loss += loss_train.detach().item()
            train_epoch_acc += acc_train.detach().item()
        
            # Gradient cropping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_train_acc = train_epoch_acc / len(train_loader)
        #print('Epoch: {:04d}'.format(epoch+1),
        #  'loss_train: {:.4f}'.format(avg_train_loss),
        #  'acc_train: {:.4f}'.format(avg_train_acc),
        #  'time: {:.4f}s'.format(time.time() - t))
        
        model.eval()
        test_loss = 0
        test_acc = 0
        test_report_label = []
        test_report_predict = []
        
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                test_data.to(args.device)
                test_label = test_data['y'].type(torch.LongTensor).to(args.device)
                test_output, test_pred = model(test_data)
        
                loss_test = criterion(test_output, test_label)
                acc_test = accuracy(test_pred, test_label)
                test_loss += loss_test.detach().item()
                test_acc += acc_test.detach().item()
        
                test_label_np = test_label.cpu().detach().numpy()
                predict = torch.max(test_pred.cpu().detach(), 1)[1]
        
                for j in range(len(test_label_np)):
                    test_report_label.append(test_label_np[j])
                    test_report_predict.append(predict[j])
            
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)
    
        report = classification_report(test_report_label, test_report_predict, output_dict = True)
        F1 = float(report["macro avg"]["f1-score"])
        
        #print("Test set results:",
        #"loss= {:.4f}".format(avg_test_loss),
        #"accuracy= {:.4f}".format(avg_test_acc),
        #"F1= {:.4f}".format(F1))
        
    torch.save(state, os.path.join(save_path, fold+'-'+args.model_name+'-'+str(max_acc)+'-'+str(max_f1)+'.th'))
        
    return avg_test_acc, F1

if __name__ == '__main__':
    seed = 22
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    datapath = '/workspace/rumor/task3/data/Pheme9_use/all-rnr-annotated-threads'
    save_path = "/workspace/rumor/task3/output/pheme/my_model"
    result_save_path = "/workspace/rumor/task3/output/pheme/my_model/result"
    fold9_data_path = '/workspace/rumor/task3/data/Pheme9_use/tfidf/'
    all_datapath = '/workspace/rumor/task3/data/Pheme9_use/all_tfidf/'
    treeLenDic = treeLenPheme(datapath)
    
    all_acc = []
    all_f1 = []
    
    #print(args)
    #sys.exit()
    for itera in range(args.iterations):
        fold0_train, fold0_test,\
        fold1_train, fold1_test,\
        fold2_train, fold2_test,\
        fold3_train, fold3_test,\
        fold4_train, fold4_test,\
        fold5_train, fold5_test,\
        fold6_train, fold6_test,\
        fold7_train, fold7_test,\
        fold8_train, fold8_test = get9folddata(fold9_data_path)
        
        acc0, f0 = train('fold0', args, all_datapath, treeLenDic, save_path, fold0_train, fold0_test)
        acc1, f1 = train('fold1', args, all_datapath, treeLenDic, save_path, fold1_train, fold1_test)
        acc2, f2 = train('fold2', args, all_datapath, treeLenDic, save_path, fold2_train, fold2_test)
        acc3, f3 = train('fold3', args, all_datapath, treeLenDic, save_path, fold3_train, fold3_test)
        acc4, f4 = train('fold4', args, all_datapath, treeLenDic, save_path, fold4_train, fold4_test)
        acc5, f5 = train('fold5', args, all_datapath, treeLenDic, save_path, fold5_train, fold5_test)
        acc6, f6 = train('fold6', args, all_datapath, treeLenDic, save_path, fold6_train, fold6_test)
        acc7, f7 = train('fold7', args, all_datapath, treeLenDic, save_path, fold7_train, fold7_test)
        acc8, f8 = train('fold8', args, all_datapath, treeLenDic, save_path, fold8_train, fold8_test)
        
        all_acc.append((acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8)/9)
        all_f1.append((f0+f1+f2+f3+f4+f5+f6+f7+f8)/9)
        
    t = time.time()  
    para = {"modelname": args.model_name,
        "in_dim": args.in_dim,
        "hidden_dim": args.hid_dim,
        "out_dim": args.out_dim,
        "Knum": args.Knum,
        "Ks": args.Ks,
        "lower": args.lower,
        "lr": args.lr,
        "wd": args.w_decay,
        "dropedge": args.dropedge,
        "batch_size": args.batchsize,
        "max_epoch": args.max_epochs,
        "interation": args.iterations,
        "acc": sum(all_acc)/args.iterations,
        "F1": sum(all_f1)/args.iterations}
        
    print("Total_Test_Accuracy: {:.4f}|Total_Test_F1: {:.4f}".format(sum(all_acc)/args.iterations, sum(all_f1)/args.iterations))
    para_save_name = "Result-time_{}-model_{}-acc_{}-f1_{}-para.json".format(t, args.model_name, sum(all_acc)/args.iterations, sum(all_f1)/args.iterations)
    with open(os.path.join(result_save_path, para_save_name),'w',encoding='utf-8') as json_file:
        json.dump(para, json_file, ensure_ascii=False)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    