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


def get_output(model, x,y, tf=True):
    # model.eval()
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
    fin_outp = torch.sigmoid(fin_outp)>0.7
    return fin_outp


def load_model(path):
    checkpoint = torch.load(path)
    model = unet(in_channels= 2, n_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_img(partition, path, idx):
    train_data = ClothDataset(root_dir = path+"/" +str(partition), use_transform=False)
    img = train_data[idx]
    x = img["X"].to(device).unsqueeze(0)
    y = img["Y"].to(device).unsqueeze(0)
    y =  torch.cat([y, torch.ones(x.shape).to(device)] , dim = 1).to(device)
    rgb = img["rgb"].to(device).permute(1,2,0)
    return x,y, rgb


if(__name__ == "__main__"):
    x,y, rgb = get_img("val", "/media/YertleDrive4/layer_grasp/dataset/2cloth_rec", 412)
    model = load_model("/home/sashank/deepl_project/cloth-segmentation/train_runs/frosty-surf-83/ckpt_latest")
    outp = get_output(model, x,y, tf=False)
    make_plot(outp.detach().cpu(), y.detach().cpu(), rgb.detach().cpu(), x.detach().cpu(), savefig="test")
    pass