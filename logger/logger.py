import logging
import logging.config
from pathlib import Path
from utils import read_json
import wandb


def train_logger(epoch, loss, auc_dict):
    # Where the magic happens
    log_dict = {**{"epoch": epoch, "loss": loss}, **auc_dict}
    wandb.log(log_dict)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
