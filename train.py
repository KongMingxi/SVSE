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
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils import setup_logger, set_random_seed, collect_env_info
from utils import clean_cfg, get_cfg_default
from config import setup_cfg


wandb.login(key='ea821429fa43e94f5a31aa9b1cf0b0da2e3d4115')


def wandb_init(cfg: DictConfig):
    wandb.init(
        project='paper4',
        group=cfg.exp_group,
        notes=cfg.exp_desc+time.strftime("%Y_%m%d_%H%M%S"),
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(cfg.ckpt_dir, 'conf.yaml'))。


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
    cfg = OmegaConf.to_yaml(cfg)
    pretty.install()
    print(OmegaConf.to_yaml(cfg))

    # setup logger
    #logger = config.get_logger('train')
    wdb_run = wandb_init(cfg)

    # setup data_loader instances
    train_data_loader, valid_data_loader = build_data_loader_training(cfg)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model = build_model_training(cfg)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      logger=wdb_run)

    trainer.train()

    wandb.finish()


if __name__ == '__main__':
    main()
