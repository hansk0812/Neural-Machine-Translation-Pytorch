import glob
import os

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("epoch_to_del_from", type=int, help="Delete all chkpts after this number")
args = ap.parse_args()

nothing = 1
for fname in glob.glob("trained_models/*"):
    epoch = int(fname.split('_epoch_')[-1].split('.pt')[0])
    
    if epoch > args.epoch_to_del_from:
        nothing = 0
        os.remove(fname)
        print ("Deleting epoch %d" % epoch, end='\r')

if nothing:
    print ("No files to delete after epoch %d!" % args.epoch_to_del_from)
