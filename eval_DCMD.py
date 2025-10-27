# eval_DCMD.py

import argparse
import os
import glob
import pytorch_lightning as pl
import yaml
from models.dcmd import DCMD
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader

if __name__== '__main__':
    
    # --- THAY ĐỔI 1: THÊM ARGUMENT MỚI CHO CHECKPOINT ---
    parser = argparse.ArgumentParser(description='DCMD Evaluation')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the config file (.yaml)')
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default=None,
                        help='(Optional) Path to a specific checkpoint file (.ckpt) to use for evaluation. Overrides automatic lookup.')
    
    # Parse args và load config
    cli_args = parser.parse_args()
    config_args = yaml.load(open(cli_args.config), Loader=yaml.FullLoader)
    config_args = argparse.Namespace(**config_args)
    config_args = init_args(config_args)
    # ---------------------------------------------------------

    # Initialize the model
    model = DCMD(config_args)
    
    if config_args.load_tensors:
        model.test_on_saved_tensors(split_name=config_args.split)
    else:
        print('Loading data and creating loaders.....')
        
        # --- THAY ĐỔI 2: LOGIC LỰA CHỌN CHECKPOINT LINH HOẠT ---
        # Ưu tiên checkpoint được cung cấp từ dòng lệnh
        if cli_args.checkpoint_path:
            ckpt_path = cli_args.checkpoint_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found at the specified path: {ckpt_path}")
            print(f"Using user-specified checkpoint: {ckpt_path}")
        # Nếu không, tự động tìm checkpoint mới nhất trong thư mục config
        else:
            print(f"No specific checkpoint path provided. Searching for the latest in: {config_args.ckpt_dir}")
            list_of_files = glob.glob(os.path.join(config_args.ckpt_dir, '*.ckpt')) 
            if not list_of_files:
                raise FileNotFoundError(f"No checkpoint files found in {config_args.ckpt_dir}. Please specify a path using -ckpt.")
            ckpt_path = max(list_of_files, key=os.path.getctime)
            print(f"Automatically selected the latest checkpoint: {ckpt_path}")
        # ---------------------------------------------------------
            
        dataset, loader, _, _ = get_dataset_and_loader(config_args, split=config_args.split)
        
        limit_test_batches = config_args.limit_test_batches if hasattr(config_args, 'limit_test_batches') else 1.0

        trainer = pl.Trainer(accelerator=config_args.accelerator, devices=config_args.devices[:1],
                             default_root_dir=config_args.ckpt_dir, max_epochs=1, logger=False,
                             limit_test_batches=limit_test_batches) 
                             
        out = trainer.test(model, dataloaders=loader, ckpt_path=ckpt_path)