import argparse
import os

import pytorch_lightning as pl
import yaml
from models.dcmd import DCMD
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader



if __name__== '__main__':
    
    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='DCMD')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    args = init_args(args)

    # Initialize the model
    model = DCMD(args)
    
    if args.load_tensors:
        # Load tensors and test
        model.test_on_saved_tensors(split_name=args.split)
    else:
        # Load test data
        print('Loading data and creating loaders.....')
        ckpt_path = '/kaggle/input/checkpoint/kaggle/working/checkpoints/HR-Avenue/train_experiment/epoch=20-step=3885.ckpt'
        dataset, loader, _, _ = get_dataset_and_loader(args, split=args.split)
        
        # Initialize trainer and test
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices[:1],
                             default_root_dir=args.ckpt_dir, max_epochs=1, logger=False,
                             # limit_test_batches=1 # để test nhanh 2 batch kiểm lỗi
                             ) 
                             
        out = trainer.test(model, dataloaders=loader, ckpt_path=ckpt_path)
