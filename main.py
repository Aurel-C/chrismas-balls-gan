import numpy as np
import argparse
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train options")
    parser.add_argument('--path',type=str, default = "noel.npy",help='Path to data')
    parser.add_argument('--framework',type=str, default='torch',help="Frawork to use (tf, torch...)")
    parser.add_argument('--epoch',type=int, default=21,help="Number of epochs to run")
    parser.add_argument('--batch_size',type=int, default=32,help="Number of image per Batch")
    args = parser.parse_args()

    data = np.load(args.path)

    if args.framework == 'tf':
        from tf_gan import train
    if args.framework == 'torch':
        from torch_gan import train
    
    wandb.login(key="01aa83e8465cdcf76446b03a37a6bd2133d07c33")
    train(data,args.epoch,args.batch_size)
