import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils import setup_logger, set_random_seed, collect_env_info
from utils import clean_cfg, get_cfg_default


def main(args):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
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
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    
    parser.add_argument('-dr', "--dataset_root",
                        type=str, default="", help="root path to dataset")
    parser.add_argument('-od', "--output_dir",
                        type=str, default="", help="output directory")

    parser.add_argument('-tr', "--trainer",
                        type=str, default="",
                        help="name of trainer")
    parser.add_argument('-mc', "--method_config_file",
                        type=str, default="",
                        help="path to config file for methods")
    parser.add_argument('-dc', "--dataset_config_file",
                        type=str, default="",
                        help="path to config file for datasets")
    parser.add_argument('-sd', "--seed",
                        type=int, default=-1,
                        help="only positive value enables a fixed seed, such as 42")
    parser.add_argument('-dv', '--device',
                        type=str, default=None,
                        help='indices of GPUs to enable (default: all)')

    parser.add_argument('-rs', "--resume",
                        type=str, default="",
                        help="checkpoint directory (from which the training resumes)")

    args = parser.parse_args()

    main(args)
