import matplotlib.pyplot as plt
import numpy as np
import os
def plot_idx(idx, root, savefig=None):
	rgb_img = np.load(root+"/rgb/"+str(img_num)+".npy")/255.0
	depth_img = np.load(root+"/depth/"+str(img_num)+".npy")
	imgs = {}
	imgs["orig"] = rgb_img[... ,::-1] #Converts BGR to RGB
	imgs["depth"] = depth_img
	mask_dir = root+"/masks"
	maskdirs = os.listdir(mask_dir)
	for i in range(len(maskdirs)):
		img =  np.load(root+"/masks/"+maskdirs[i]+"/"+str(img_num)+".npy")
		imgs[str(maskdirs[i])] = img
	rows = 1
	cols = len(imgs.keys())
	fig = plt.figure()
	ax = []
	for i in range(cols*rows):
		name = list(imgs.keys())[i]
		ax.append( fig.add_subplot(rows, cols, i+1) )
		ax[-1].set_title(name)  # set title
		plt.imshow(imgs[name])
	if(savefig == None):
		plt.show()
	else:
		plt.savefig(savefig, bbox_inches='tight')
	return None

def plot_sample(sample, savefig=None):
	rgb_img = sample['rgb'].permute(1,2,0).numpy()
	depth_img = sample['X'][0, :, :].numpy()
	imgs = {}
	imgs["orig"] = rgb_img[..., ::-1] #Converts BGR to RGB
	print(imgs["orig"].shape)
	imgs["depth"] = depth_img
	for i in range(sample['Y'].shape[0]):
		imgs[str(i)] = sample['Y'][i,:,:].numpy()
	rows = 1
	cols = len(imgs.keys())
	fig = plt.figure()
	ax = []
	for i in range(cols*rows):
		name = list(imgs.keys())[i]
		ax.append( fig.add_subplot(rows, cols, i+1) )
		ax[-1].set_title(name)  # set title
		plt.imshow(imgs[name])
	if(savefig == None):
		plt.show()
	else:
		plt.savefig(savefig, bbox_inches='tight')
	return None

def make_plot(outp, y, rgb, x, savefig=None):
	cols = 5
	rows = 3
	fig = plt.figure()
	fig.tight_layout()
	ax = []
	for i in range(cols*rows):
		ax.append(fig.add_subplot(rows, cols, i+1) )
		if(i%5 == 0):
			if(i<5):
				ax[-1].set_title("RGB Image")
			plt.imshow(rgb[:,:,:])
		if(i%5 == 1):
			if(i<5):
				ax[-1].set_title("Depth Image")
			plt.imshow(x[0,0,:,:])
		if(i%5 == 2):
			if(i<5):
				ax[-1].set_title("Input Mask")
				plt.imshow(np.zeros(rgb.shape[:-1]))
			else:
				plt.imshow(outp[0,int(i/5) -1,:,:])
		if(i%5 == 3):
			if(i<5):
				ax[-1].set_title("Output Mask")
			plt.imshow(outp[0,int(i/5),:,:])
		if(i%5 == 4):
			if(i<5):
				ax[-1].set_title("Desired Mask")
			plt.imshow(y[0,int(i/5),:,:])
	if(savefig == None):
		plt.savefig("inp_outp_plot")
		plt.close()
	else:
		plt.savefig(savefig)
		plt.close()

def plot_network():
	pass
if(__name__ == "__main__"):
	path = "/media/YertleDrive4/layer_grasp/dataset/2cloth_rec/test"
	img_num = 166
	from data_loader import RecClothDataset
	train_data = RecClothDataset(root_dir="/media/YertleDrive4/layer_grasp/dataset/2cloth_rec/train", num_masks=2, use_transform=True)
	sample = train_data[img_num]
	plot_sample(sample)