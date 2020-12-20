import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train options")
    parser.add_argument('--path',type=str, default = "noel.npy",help='Path to data')
    parser.add_argument('--framework',type=str, default='torch',help="Frawork to use (tf, torch...)")
    parser.add_argument('--epoch',type=int, default=21,help="Number of epochs to run")
    args = parser.parse_args()

    data = np.load(args.path)

    if args.framework == 'tf':
        from tf_gan import train
    if args.framework == 'torch':
        from torch_gan import train
    
    train(data,args.epoch)
