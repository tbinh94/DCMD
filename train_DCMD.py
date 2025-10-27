# train_DCMD.py

import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from models.dcmd import DCMD
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader
from utils.ema import EMACallback


if __name__== '__main__':

    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the config file (.yaml)')
    
    cli_args = parser.parse_args()
    config_path = cli_args.config
    args = yaml.load(open(config_path), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    args = init_args(args)
    # Save a copy of the config file to the checkpoint directory for reproducibility
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.system(f'cp {config_path} {os.path.join(args.ckpt_dir, "config.yaml")}')     
    
    # Set seeds for reproducibility 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 
    pl.seed_everything(args.seed)

    # Set callbacks and logger
    if (hasattr(args, 'diffusion_on_latent') and args.stage == 'pretrain'):
        monitored_metric = 'pretrain_rec_loss'
        metric_mode = 'min'
    elif args.validation:
        monitored_metric = 'AUC'
        metric_mode = 'max'
    else:
        # For datasets without validation like Avenue, monitor a training loss
        monitored_metric = 'loss1' # Changed from 'pre_loss' to a more relevant logged metric
        metric_mode = 'min'
    
    print(f"Monitoring metric: '{monitored_metric}' in '{metric_mode}' mode for ModelCheckpoint.")
        
    callbacks = [ModelCheckpoint(
                    dirpath=args.ckpt_dir,
                    save_top_k=2,
                    save_last=True,
                    monitor=monitored_metric,
                    mode=metric_mode,
                    filename='{epoch}-{step}-{' + monitored_metric + ':.2f}' # Improved filename
                )]
    
    # --- TỰ ĐỘNG KÍCH HOẠT SWA CHO CÁC LẦN HUẤN LUYỆN DÀI ---
    # SWA is most effective with longer training runs and a cosine scheduler.
    if args.n_epochs > 50: # A reasonable threshold to start using SWA
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-4)) # Adjusted lr for SWA
        print("SWA is enabled for this long training run.")
    else:
        print("SWA is disabled for this short test run (n_epochs <= 50).")
    # --------------------------------------------------------
    
    callbacks += [EMACallback()] if args.use_ema else []
    
    if args.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        wandb_logger = WandbLogger(project=args.project_name, group=args.group_name, entity=args.wandb_entity, 
                                   name=args.dir_name, config=vars(args), log_model='all')
    else:
        wandb_logger = None # Use None instead of False for clarity

    # Get dataset and loaders
    _, train_loader, _, val_loader = get_dataset_and_loader(args, split=args.split, validation=args.validation)

    # For OneCycleLR optimizer scheduler
    steps_per_epoch = len(train_loader)
    
    # Initialize model
    model = DCMD(args, steps_per_epoch=steps_per_epoch)
    
    # --- SỬ DỤNG limit_train_batches TỪ FILE CONFIG ---
    # Set default to 1.0 (use all batches) if not specified in the config file.
    limit_train_batches = args.limit_train_batches if hasattr(args, 'limit_train_batches') else 1.0
    if limit_train_batches < 1.0:
        print(f"Limiting training to {limit_train_batches*100:.0f}% of batches per epoch.")
    # ----------------------------------------------------
    
    trainer = pl.Trainer(accelerator=args.accelerator, 
                         devices=args.devices, 
                         default_root_dir=args.ckpt_dir, 
                         max_epochs=args.n_epochs, 
                         logger=wandb_logger, 
                         callbacks=callbacks, 
                         strategy=DDPStrategy(find_unused_parameters=False),
                         log_every_n_steps=20, 
                         num_sanity_val_steps=0, 
                         deterministic=True, 
                         limit_train_batches=limit_train_batches
                         )
    
    # Train the model    
    print("Starting training...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")