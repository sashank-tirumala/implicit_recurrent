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
from unet import unet
from data_loader import RecClothDataset as ClothDataset
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
from utils import weights_init, compute_map, compute_iou, compute_auc, preprocessHeatMap 
from plot_utils import make_plot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def train_mixed_prec(model, train_loader, criterion, optimizer, scheduler, scaler, i_ini,  using_wandb=False, tf=True, epoch=0):
# 	model.train()
# 	total_loss = 0
# 	for i, samples in enumerate(train_loader):
# 		with torch.cuda.amp.autocast():  
# 			x = samples['X'].to(device)
# 			y = samples['Y'].to(device)
# 			outp = model.forward(x, outp=y, mode="train", teacher_forcing=tf)
# 			loss = criterion(outp, y)
			
# 		total_loss += float(loss)
# 		optimizer.zero_grad()
# 		scaler.scale(loss).backward()
# 		scaler.step(optimizer)
# 		scaler.update()
# 		if(using_wandb):
# 			wandb.log({"loss":float(total_loss / (i + 1)), "step":int(i_ini), 'lr': float(optimizer.param_groups[0]['lr'])})
# 		if(scheduler is not None):
# 			scheduler.step()
# 		i_ini += 1
# 	return i_ini, float(total_loss / (i + 1))

def recurrent_train(model, train_loader, criterion, optimizer, scheduler, i_ini,  using_wandb=False, tf=True, epoch=0):
	model.train()
	total_loss = 0
	ious = []
	count = 0
	torch.cuda.empty_cache()
	for itercount, samples in enumerate(train_loader):  
		x = samples['X'].to(device)
		y = samples['Y'].to(device)
		fin_outp = torch.zeros(x.shape).to(device)
		for i in range(y.shape[1]+1):
			if tf:
				if(i == 0):
					cur_inp = torch.cat([x, fin_outp] , dim = 1).to(device)
				else:
					cur_inp = torch.cat([x,y[:,i-1,:,:].unsqueeze(1)], dim=1).to(device)
			else:
				if(i == 0):
					cur_inp = torch.cat([x, fin_outp] , dim = 1).to(device)
				else:
					cur_inp = torch.cat([x,(torch.sigmoid(outp)>0.7).to(torch.float)], dim=1).to(device)
			outp = model(cur_inp)
			fin_outp = torch.cat([fin_outp, outp] , dim = 1).to(device)
		fin_outp = fin_outp[:,1:,:,:]
		target =  torch.cat([y, torch.ones(x.shape).to(device)] , dim = 1).to(device)
		loss = criterion(fin_outp, target)	
		total_loss += float(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if(using_wandb):
			wandb.log({"loss":float(total_loss / (itercount + 1)), "step":int(i_ini), 'lr': float(optimizer.param_groups[0]['lr']), 'epoch':epoch})
		if(scheduler is not None):
			scheduler.step()
		i_ini += 1
		batch_iou = metrics(fin_outp, target)
		ious = ious + batch_iou
		if(epoch%10 == 0 and count == 0):
			rgb = samples['rgb'].permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
			rgb = rgb[... , ::-1]
			fin_outp = (torch.sigmoid(fin_outp)>0.7).to(torch.float)
			make_plot(fin_outp.detach().cpu(), target.detach().cpu(), rgb, x.detach().cpu(), savefig="imgs/rec_train")
			wandb.log({"train_viz": wandb.Image("imgs/rec_train.png")})
			count +=1
	ious = np.nanmean(ious)
	if(using_wandb):
		wandb.log({"training_iou":ious})
	return i_ini, float(total_loss / (itercount + 1))

def validate(model, val_loader, criterion,  using_wandb=False, epoch=0):
	# model.eval()
	val_loss = 0
	ious = []
	count = 0
	torch.cuda.empty_cache()
	for itercount, samples in enumerate(val_loader):  
		with torch.no_grad():
			x = samples['X'].to(device)
			y = samples['Y'].to(device)
			fin_outp = torch.zeros(x.shape).to(device)
			for i in range(y.shape[1]+1):
				if(i == 0):
					cur_inp = torch.cat([x, fin_outp] , dim = 1).to(device)
				else:
					cur_inp = torch.cat([x,(torch.sigmoid(outp)>0.7).to(torch.float)], dim=1).to(device)
				outp = model(cur_inp)
				fin_outp = torch.cat([fin_outp, outp] , dim = 1).to(device)

			fin_outp = fin_outp[:,1:,:,:]
			target =  torch.cat([y, torch.ones(x.shape).to(device)] , dim = 1).to(device)
			loss = criterion(fin_outp, target)
			batch_iou = metrics(fin_outp, target)
			val_loss += float(loss)
			ious = ious + batch_iou
			if(using_wandb):
				wandb.log({"val_loss":float(val_loss / (itercount + 1))})
		if(epoch%10 == 0 and count == 0):
			rgb = samples['rgb'].permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
			rgb = rgb[... , ::-1]
			make_plot(torch.sigmoid(fin_outp.detach().cpu()), target.detach().cpu(), rgb, x.detach().cpu(), savefig="imgs/rec_val")
			wandb.log({"val_viz": wandb.Image("imgs/rec_val.png")})
			count +=1
	ious = np.nanmean(ious)
	if(using_wandb):
		wandb.log({"val_iou":ious,  "epoch":epoch})
	return float(val_loss / (itercount + 1))

def metrics(outputs, labels):
	output = torch.sigmoid(outputs[:,:,:,:])
	output = output.data.cpu().numpy()
	pred = output.transpose(0, 2, 3, 1)
	gt = labels.cpu().numpy().transpose(0, 2, 3, 1)
	ious = []
	aucs = []
	for g, p in zip(gt, pred):
		ious.append(compute_iou(g, p, 3))
		# aucs.append(compute_auc(g, p, 2))
	return ious

def get_dataloaders(cfg):
	train_data = ClothDataset(root_dir = cfg["datapath"]+"/train", use_transform=True) #Was false, could be why it was not generalizing
	val_data = ClothDataset(root_dir = cfg["datapath"] + "/val", use_transform=False)
	train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
	val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=True)
	return train_loader, val_loader
def get_teacher_forcing(e, cfg):
	num = np.random.random()
	if(num < cfg["teacher_forcing"]):
		return True
	else:
		return False
def training(cfg):
	train_loader, val_loader = get_dataloaders(cfg)
	if(cfg["model_path"] is None):
		model = unet(in_channels= 2, n_classes=1, is_batchnorm=True).to(device)
	else:
		model = load_model(model_path).to(device)
	optimizer = optim.Adam(model.parameters(), lr = cfg["lr"])
	scheduler = None
	if(cfg["scheduler"]):
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*cfg["epochs"], eta_min=1e-6, last_epoch=- 1, verbose=False)
	criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to(device), reduce='sum')
	i_ini = 0
	val_loss= np.array([])
	for e in range(cfg["epochs"]):
		start = time.time()
		i_ini, loss = recurrent_train(model, train_loader, criterion, optimizer, scheduler, i_ini,  using_wandb=cfg["wandb"], tf=get_teacher_forcing(e, cfg), epoch=e)
		cur_val_loss = validate(model, val_loader, criterion, using_wandb=cfg["wandb"], epoch=e)
		val_loss = np.append(val_loss, cur_val_loss)

		stop = time.time()
		if(cfg["wandb"]):
			wandb.log({"epoch_time":(stop-start)/60.0})
		rank = (val_loss < cur_val_loss).sum()
		# print(cur_val_loss, val_loss, rank)
		save_model(model, optimizer, scheduler, loss,  cfg, e, rank = rank + 1)

def save_model(model, optimizer, scheduler, loss,  cfg, epoch, rank=10):
	torch.save({'epoch': epoch, 
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict':optimizer.state_dict(),
			'loss':loss,
			'cfg':cfg
			}, cfg["runspath"]+"/"+"ckpt_latest")
	if(rank < 6.0):
		torch.save({'epoch': epoch, 
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'loss':loss,
				'cfg':cfg
				}, cfg["runspath"]+"/"+"ckpt_"+str(rank))

def load_model():
	pass
if __name__ == '__main__':
	torch.manual_seed(1337)
	torch.cuda.manual_seed(1337)
	np.random.seed(1337)
	random.seed(1337)
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-lr','--lr', type=float, help='learning rate', default = 1e-3) 
	parser.add_argument('-wd','--w_decay', type=float, help='weight decay (regularization)', default=0) 
	parser.add_argument('-m','--momentum', type=float, help='momentum (Adam)',  default=0)
	parser.add_argument('-bs','--batch_size', type=int, help='Description for bar argument', default=8)
	parser.add_argument('-e','--epochs', type=int, help='Number of epochs to train for', default=50)
	parser.add_argument('-dp','--datapath', type=str, help='Where is the dataset stored', default="/home/sashank/deepl_project/data/dataset/test/")
	parser.add_argument('-rp','--runspath', type=str, help='Where to store runs data', default="/home/sashank/deepl_project/cloth-segmentation/train_runs")
	parser.add_argument('-nc','--n_class', type=int, help='Number of masks to predict', default=2)
	parser.add_argument('-nf','--n_feature', type=int, help='Number of input masks to predict', default=2)
	parser.add_argument('-wandb','--wandb', type=int, help='use wandb or not', default=1)
	parser.add_argument('-tf','--teacher_forcing', type=float, help='teacher_forcing', default=0.5)
	parser.add_argument('-mp','--model_path', type=str, help='train from existing model', default=None)
	parser.add_argument('-sch','--scheduler', type=int, help='Use lr-scheduler', default=0)

	args = vars(parser.parse_args())

	if args['wandb']:
		run = wandb.init(project="CORL2022", entity="stirumal", config=args)
		args["runspath"] = args["runspath"]+"/"+run.name
		os.makedirs(args["runspath"])
	training(args)
	