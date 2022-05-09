import numpy as np 
import os
import shutil 
import argparse
def create_dirs(root, del_if_exists = True):
    if os.path.isdir(root):
        if del_if_exists:
            print("Is already a directory, deleting and creating again")
            shutil.rmtree(root)
        else:
            print("Is already a directory, quitting program")
            exit()
    os.makedirs(root)
    dirs = ["train", "val", "test"]
    for cur_dir in dirs:
        new_dir = ["rgb", "depth", "masks"]
        for nd in new_dir:
            os.makedirs(root+"/"+cur_dir+"/"+nd)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Creates required dataset for rec format')
    parser.add_argument('-nn','--new_dir_name', type=str, help='Path of new dataset', default = "crazy-life")
    args = vars(parser.parse_args())
    create_dirs(args["new_dir_name"], del_if_exists=False)
