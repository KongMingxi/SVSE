import argparse
import time
import collections
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from rich import pretty, print

#import data_loader.data_loaders as module_data
from data_loader import build_data_loader_training
import modeling.loss as bce_loss
from modeling.optimizer import build_optimizer
from modeling.metric import CxrEvaluator
from modeling.network import build_model_training

from trainer import Trainer
from utils import prepare_device
from utils import setup_logger, set_random_seed, collect_env_info
from utils import clean_cfg, get_cfg_default
from config import setup_cfg


wandb.login(key='ea821429fa43e94f5a31aa9b1cf0b0da2e3d4115')

def wandb_init(cfg: Dict):
    wandb.init(
        project='paper4',
        group=cfg.exp_group,
        notes=cfg.exp_desc+time.strftime("%Y_%m%d_%H%M%S"),
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(cfg.ckpt_dir, 'conf.yaml'))


@hydra.main(version_base=None, config_path="./config",  config_name="default_config")
def main(cfg: DictConfig) -> None:
    '''
    1. setup configs using Hydra and Omegaconf
    2. setup logger using Wandb
        what are written into logger:
        (1) trian loss (value), train auc (value) [per class, average]
        (2) val loss (value), val auc (value) [per class, average]
        (3) ROC curves (plot) [per class, average]
    3. build dataloaders for training
    4. build model, loss, optimizer, etc
        To build a model:
        (1) network (backbone, classifier)
        (2) loss
        (3) optimizer
        (4) lr scheduler
    '''
    # setup config obj.
    #cfg = setup_cfg(args)
    pretty.install()   
    print(OmegaConf.to_yaml(cfg))
    cfgs = OmegaConf.to_container(cfg, resolve=True)
 
    # setup logger
    #logger = config.get_logger('train')
    wandb_init(cfgs)

    # setup data_loader instances
    train_data_loader, valid_data_loader = build_data_loader_training(cfgs)

    # build model architecture, then print to console
    model = build_model_training(cfgs)
    #logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = build_loss(cfgs) # bce_loss
    metrics = build_evaluator(cfgs) # CxrEvaluator

    # build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer = build_optimizer(model, cfgs)
    lr_scheduler = build_lr_scheduler(optimizer, cfgs)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=cfgs,
                      device=device,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      logger=wandb)

    trainer.train()

    wandb.finish()


if __name__ == '__main__':
    main()
