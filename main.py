import os
import argparse
import torch
from solver import Solver
from torch.backends import cudnn
from torch.utils.data import Subset
import json
import random
import numpy as np

import sys
sys.path.append('../..')
sys.path.append('/home/bruno/xfang/GenrativeMethod/model/pytorch3dunet')
from CTDataset import StrokeAI

def str2bool(v):
    return v.lower() in ('true')

def save_indices(indices, file_path):
    with open(file_path, 'w') as file:
        json.dump(indices, file)

def load_indices(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def train_test_split(dataset, test_size=0.2, random_seed=42, indices_file=None):
    """
    Make sure the reproductibility and consistence of each experiment: Test set remains the same
    
    Args: 
    indices_file: indices for test dataset
    
    Returns:
    Same train dataset and test dataset for each run
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Check if indices file exists
    if indices_file and os.path.exists(indices_file):
        print("load fixed dataset split")
        indices = load_indices(indices_file)
    else:
        # Generate a shuffled list of indices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        save_indices(indices, indices_file if indices_file else 'dataset_indices.json')

    split = int(np.floor(test_size * len(indices)))
    train_indices, test_indices = indices[split:], indices[:split]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    dataset = StrokeAI(CT_root="/home/bruno/xfang/dataset/images",
                       MRI_root="/scratch4/rsteven1/DWI_coregis_20231208", 
                       label_root="/home/bruno/xfang/dataset/labels", 
                       bounding_box=True,
                       instance_normalize=True, 
                       padding=True, 
                    #    slicing=True,
                       crop=True)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42, 
                                                   indices_file='/home/bruno/3D-Laision-Seg/GenrativeMethod/dataset_indices.json')
    dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=not config.serial_batches,
            num_workers=int(config.num_workers))
    

    # Solver for training and testing StarGAN.
    solver = Solver(dataloader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')  
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')  
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')  
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')  
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')  
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')  
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer') 
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')  
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')  
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=135000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])  
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)  
    
    parser.add_argument('--log_dir', type=str, default='CT2MRI_3DGAN/logs') 
    parser.add_argument('--model_save_dir', type=str, default='CT2MRI_3DGAN/models') 
    parser.add_argument('--sample_dir', type=str, default='CT2MRI_3DGAN/samples') 
    parser.add_argument('--result_dir', type=str, default='CT2MRI_3DGAN/results') 

    # Step size.
    parser.add_argument('--log_step', type=int, default=10) 
    parser.add_argument('--sample_step', type=int, default=5000) 
    parser.add_argument('--model_save_step', type=int, default=5000) 
    parser.add_argument('--lr_update_step', type=int, default=1000) 

    config = parser.parse_args()
    print(config)
    main(config)