import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train options")
    parser.add_argument('--path',type=str, default = "noel.npy",help='Path to data')
    parser.add_argument('--framework',type=str, default='torch',help="Frawork to use (tf, torch...)")
    args = parser.parse_args()

    data = np.load(args.path)

    if args.framework == 'tf':
        from tf_gan import train
    if args.framework == 'torch':
        from torch_gan import train
    
    train(data)
