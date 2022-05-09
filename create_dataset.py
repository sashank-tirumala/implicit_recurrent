import numpy as np 
import os
import shutil 
import argparse
from alive_progress import alive_bar
def create_dirs(root, base_root, del_if_exists):
    if os.path.isdir(root):
        if del_if_exists:
            print("Is already a directory, deleting and creating again")
            shutil.rmtree(root)
        else:
            print("Is already a directory, quitting program")
            exit()
    mask_dirs = ["masks/" + y for y in os.listdir(base_root+"/masks")]
    os.makedirs(root)
    dirs = ["train", "val", "test"]
    for cur_dir in dirs:
        new_dir = ["rgb", "depth"] + mask_dirs
        for nd in new_dir:
            os.makedirs(root+"/"+cur_dir+"/"+nd)

def add_images(target_root, base_root, img_paths):
    mask_dirs = ["masks/" + y for y in os.listdir(base_root+"/masks")]
    partitions = ["rgb", "depth"]+mask_dirs
    count = 0
    with alive_bar(len(img_paths)) as bar:
        for i in range(len(img_paths)):
            for dirn in partitions:
                numpy_img = np.load(base_root+"/"+dirn+"/"+img_paths[i])
                np.save(target_root+"/"+dirn+"/"+str(count)+".npy", numpy_img)
            count +=1
            bar()
def create_dataset(base_path, target_path, del_if_exists = True, split=[0.7, 0.2, 0.1]):
    create_dirs(target_path,base_path, del_if_exists)
    rgb_img_path = np.array(os.listdir(base_path+"/rgb"))
    np.random.shuffle(rgb_img_path)
    print("Train: Val: Test split ", split)
    N = rgb_img_path.shape[0]
    train_paths = rgb_img_path[:int(split[0]*N)]
    val_paths = rgb_img_path[int((split[0])*N):int((split[0]+split[1])*N)]
    test_paths = rgb_img_path[int((split[0]+split[1])*N):]

    print("Working on creating Train Dataset!")
    add_images(target_path+"/train", base_path, train_paths)
    print("Working on creating Val Dataset!")
    add_images(target_path+"/val", base_path, val_paths)
    print("Working on creating Test Dataset!")
    add_images(target_path+"/test", base_path, test_paths)
    print("Created Dataset!")

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Creates required dataset for rec format')
    parser.add_argument('-bn','--base_path', type=str, help='Path of base dataset', default = "driven-planet")
    parser.add_argument('-nn','--target_path', type=str, help='Path of new dataset', default = "crazy-life")
    args = vars(parser.parse_args())
    # create_dirs(args["new_dir_name"], del_if_exists=False)
    create_dataset(**args)
