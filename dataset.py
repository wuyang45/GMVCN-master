#coding: utf8
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data

seed = 22
np.random.seed(seed)
random.seed(seed)

class GraphDataset(Dataset):
    def __init__(self, root, file_list, treeLenDic, dropedge=0, lower = 2, upper = 100000):
        super(GraphDataset, self).__init__()
        
        self.root = root
        self.file_list = list(filter(lambda id: id.split('.')[0] in treeLenDic.keys() and treeLenDic[id.split('.')[0]] >= lower and treeLenDic[id.split('.')[0]] <= upper, file_list))
        self.dropedge = dropedge
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        id = self.file_list[idx]
        
        data = np.load(os.path.join(self.root, id), allow_pickle=True)
        #data['x'] = data['x'].astype(float)
        edgeindex = data['edge_index']
        BU_edgeindex = data['BU_edge_index']
        if self.dropedge > 0:
            row = edgeindex[0]
            col = edgeindex[1]
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.dropedge)))
            poslist = sorted(poslist)
            row = np.array(row)[poslist]
            col = np.array(col)[poslist]
            new_edgeindex = np.array([row, col])
            #print('TD:{}'.format(new_edgeindex.shape))
            new_edgeindex1 = np.array([col, row])
            
            udirIndex = np.concatenate((new_edgeindex, new_edgeindex1), axis=1)
            #print('udir:{}'.format(udirIndex.shape))
            
            BUrow = BU_edgeindex[0]
            BUcol = BU_edgeindex[1]
            BUlength = len(BUrow)
            BUposlist = random.sample(range(BUlength), int(BUlength * (1 - self.dropedge)))
            BUposlist = sorted(BUposlist)
            BUrow = np.array(BUrow)[BUposlist]
            BUcol = np.array(BUcol)[BUposlist]
            BUnew_edgeindex = np.array([BUrow, BUcol])
            #print('BU:{}'.format(BUnew_edgeindex.shape))
            
        else:
            new_edgeindex = edgeindex
            BUnew_edgeindex = BU_edgeindex
            
            udirIndex = np.concatenate((edgeindex, BU_edgeindex), axis=1)
            
        return Data(x=torch.FloatTensor(data['x']), 
                 edge_index=torch.LongTensor(new_edgeindex),
                 BU_edge_index=torch.LongTensor(BUnew_edgeindex),
                 y=torch.LongTensor([int(data['y'])]),
                 x_user=torch.tensor(data['x_user'], dtype=torch.float32),
                 udir_edge_index=torch.LongTensor(udirIndex),
                 sid = id.split('.')[0]
                )
    