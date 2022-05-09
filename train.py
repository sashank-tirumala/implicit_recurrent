#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from recurrent_unet import recurrent_unet
from data_loader import ClothDataset
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import weights_init, compute_map, compute_iou, compute_auc, preprocessHeatMap 
# from tensorboardX import SummaryWriter
from PIL import Image
import wandb
import argparse
import torchvision.transforms as T
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_mixed_prec(model, train_loader, criterion, optimizer, scheduler, scaler, i_ini,  using_wandb=False, tf=True, epoch=0):
	model.train()
	total_loss = 0
	for i, samples in enumerate(train_loader):
		with torch.cuda.amp.autocast():  
			x = samples['X'].to(device)
			y = samples['Y'].to(device)
			outp = model.forward(x, outp=y, mode="train", teacher_forcing=tf)
			loss = criterion(outp, y)
			
		total_loss += float(loss)
		print(total_loss)
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		if(using_wandb):
			wandb.log({"loss":float(total_loss / (i + 1)), "step":int(i_ini), 'lr': float(optimizer.param_groups[0]['lr'])})
		if(scheduler is not None):
			scheduler.step()
		i_ini += 1
	return i_ini, float(total_loss / (i + 1))

def train(model, train_loader, criterion, optimizer, scheduler, i_ini,  using_wandb=False, tf=True, epoch=0):
	model.train()
	total_loss = 0
	for i, samples in enumerate(train_loader):  
		x = samples['X'].to(device)
		y = samples['Y'].to(device)
		outp = model.forward(x, outp=y, mode="train", teacher_forcing=tf)
		loss = criterion(outp, y)	
		total_loss += float(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if(using_wandb):
			wandb.log({"loss":float(total_loss / (i + 1)), "step":int(i_ini), 'lr': float(optimizer.param_groups[0]['lr'])})
		if(scheduler is not None):
			scheduler.step()
		i_ini += 1
		torch.cuda.empty_cache()
	return i_ini, float(total_loss / (i + 1))

def validate(model, val_loader, criterion,  using_wandb=False, tf=True, epoch=0):
	model.eval()
	val_loss = 0
	for i, samples in enumerate(val_loader):  
		with torch.no_grad():
			x = samples['X'].to(device)
			y = samples['Y'].to(device)
			outp = model.forward(x, outp=y, mode="train", teacher_forcing=tf)
			loss = criterion(outp, y)
			val_loss += float(loss)
			if(using_wandb):
				wandb.log({"val_loss":float(val_loss / (i + 1))})
	return float(val_loss / (i + 1))

def get_dataloaders(cfg):
	train_data = ClothDataset(root_dir = cfg["datapath"]+"/train", use_transform=False)
	val_data = ClothDataset(root_dir = cfg["datapath"] + "/val", use_transform=False)
	train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
	val_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
	return train_loader, val_loader

def training(cfg):
	train_loader, val_loader = get_dataloaders(cfg)
	model = recurrent_unet(in_channels= 2, n_classes=1).to(device)
	optimizer = optim.Adam(model.parameters(), lr = cfg["lr"])
	scheduler = None
	criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to(device), reduce='sum')
	i_ini = 0
	for e in range(cfg["epochs"]):
		start = time.time()
		i_ini, loss = train(model, train_loader, criterion, optimizer, scheduler, i_ini,  using_wandb=True, tf=True, epoch=e)
		validate(model, val_loader, criterion, using_wandb=True, tf=True, epoch=e)
		stop = time.time()
		if(cfg["wandb"]):
			wandb.log({"epoch_time":(stop-start)/60.0})
		if(e%10 == 0):
			save_model(model, optimizer, scheduler, loss,  cfg, e)

def save_model(model, optimizer, scheduler, loss,  cfg, epoch):
	if(scheduler is not None):
		torch.save({'epoch': epoch, 
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'scheduler_state_dict':scheduler.state_dict(),
				'loss':loss,
				'cfg':cfg
				}, cfg["runspath"]+"/"+"ckpt_"+str(epoch))
	else:
		torch.save({'epoch': epoch, 
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'loss':loss,
				'cfg':cfg
				}, cfg["runspath"]+"/"+"ckpt_"+str(epoch))

if __name__ == '__main__':
	torch.manual_seed(1337)
	torch.cuda.manual_seed(1337)
	np.random.seed(1337)
	random.seed(1337)
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-lr','--lr', type=float, help='learning rate', default = 1e-3) # required=True ,
	parser.add_argument('-wd','--w_decay', type=float, help='weight decay (regularization)', default=0) #required=True ,
	parser.add_argument('-m','--momentum', type=float, help='momentum (Adam)',  default=0)#required=True ,
	parser.add_argument('-ss','--step_size', type=int, help='Description for bar argument',  default=30)#required=True ,
	parser.add_argument('-g','--gamma', type=float, help='Description for foo argument', default=0.5)#required=True ,
	parser.add_argument('-bs','--batch_size', type=int, help='Description for bar argument', default=8)#required=True ,
	parser.add_argument('-e','--epochs', type=int, help='Description for bar argument', default=50)#required=True ,
	parser.add_argument('-dp','--datapath', type=str, help='Description for bar argument', default="/home/sashank/deepl_project/data/dataset/test/")#required=True ,
	parser.add_argument('-rp','--runspath', type=str, help='Description for bar argument', default="/home/sashank/deepl_project/cloth-segmentation/train_runs")#required=True ,
	parser.add_argument('-t','--transform', type=bool, help='Description for bar argument', default=False)#required=True ,
	parser.add_argument('-nc','--n_class', type=int, help='Description for bar argument', default=2)#required=True ,
	parser.add_argument('-nf','--n_feature', type=int, help='Description for bar argument', default=2)#required=True ,
	parser.add_argument('-ds','--datasize', type=str, help='Description for bar argument', default="")#required=True ,
	parser.add_argument('-wandb','--wandb', type=int, help='use wandb or not', default=1)#required=True ,

	args = vars(parser.parse_args())

	# # with open('configs/segmentation.json', 'r') as f:
	# #     cfgs = json.loads(f.read())
	# # print(json.dumps(cfgs, sort_keys=True, indent=1))
	run = wandb.init(project="CORL2022", entity="stirumal", config=args)
	args["runspath"] = args["runspath"]+"/"+run.name
	# oldmask = os.umask(000)
	os.makedirs(args["runspath"])

	training(args)
