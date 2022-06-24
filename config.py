#coding: utf8
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--in_dim', type=int, default=5000)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=64)
parser.add_argument('--Knum', type=int, default=64)
parser.add_argument('--Ks', type=list, default=[1])
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--w_decay', type=float, default=1e-3)
parser.add_argument('--dropedge', type=float, default=0)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--model_name', type=str, default='GMVCN')
parser.add_argument('--lower', type=int, default=2)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=22)

args = parser.parse_args()

args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

